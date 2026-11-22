"""Frozen step-200 forward pass and paired scoring with one source encoding for its prespecified target directions."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from fieldbridge.evaluation.decoder_context import SOURCE_NATIVE_EXTERIOR, decoder_input_latent
from fieldbridge.data.domains import Domain
from fieldbridge.data.latent_bank import encode_latent
from fieldbridge.data.photometry_factored_latent_bank import propagate_encoder_local_valid_core_support
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256, validated_source_volume
from fieldbridge.evaluation.mrixfields2026_official import official_task3_nrmse
from fieldbridge.evaluation.retrospective_bypass import finish, summarize_difference
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    _require_linear_decoder_output_activation, adapt_decoder_evaluation_range, adapt_full_volume_layout,
)
from fieldbridge.evaluation.zero_transport_diagnostic import _ieee_encoder_convolutions
from fieldbridge.training.stage2_unified import integrate_transport


@contextmanager
def _matched_rng(device, case_seed):
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.random.default_generator.manual_seed(case_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(case_seed)
        yield


def _solve(translator, start, source_domain, request, device, case_seed):
    times = []
    labels = []

    def observed_call(_module, args):
        value, source, target, time = args
        if list(source) != [source_domain] or list(target) != [request] or value.shape != start.shape:
            raise ValueError("Observed velocity call disagrees with scheduled source, request or latent shape.")
        times.append(float(time.detach().item()))
        labels.append(target[0].label)

    handle = translator.register_forward_pre_hook(observed_call)
    try:
        with _matched_rng(device, case_seed), torch.autocast(device.type, dtype=torch.bfloat16):
            endpoint = integrate_transport(translator, start, [source_domain], [request], steps=4, solver="heun")
    finally:
        handle.remove()
    if times != [0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0]:
        raise ValueError(f"Unexpected frozen Heun evaluation schedule: {times}.")
    return endpoint, times, labels


def _differences(first, second, regions):
    totals = {}
    summarize_difference(totals, "output", first[0].numpy(), second[0].numpy(), regions)
    return {key.split("/", 1)[1]: data for key, data in finish(totals).items()}


@dataclass
class PreparedSource:
    source_domain: Domain
    expected_shape: tuple
    source: torch.Tensor
    context: object
    foreground: torch.Tensor
    core: torch.Tensor
    mask: torch.Tensor
    native: torch.Tensor
    start: torch.Tensor
    start_hash: str


@torch.inference_mode()
def prepare_source(*, encoder, artifact, stats, support_rule_sha256, source, source_domain, device, mark=print):
    """Encode a frozen source once, retaining only its tensors until this source group completes."""
    device = torch.device(device)
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Frozen encoder required.")
    mark(f"prepare {source_domain.label} source and valid core")
    layout = adapt_full_volume_layout(source, role="measured source")
    expected_shape = layout.authenticated_shape
    source = validated_source_volume(layout.model_batch(), source_domain.label)
    context = artifact.normalize_source(source, source_domain)
    foreground = source != 0
    if not torch.equal(context.support_mask, foreground) or not bool(foreground.any()):
        raise ValueError("Empty or inconsistent source-derived foreground.")
    core = propagate_encoder_local_valid_core_support(
        foreground, encoder, expected_rule_sha256=support_rule_sha256)
    if not bool(core.any()) or bool(core.all()):
        raise ValueError("The source must have valid-core and exterior latent cells.")
    mask = core[None, None].to(device)

    mark("encode measured source once in FP32")
    with _ieee_encoder_convolutions(device), torch.autocast(device.type, enabled=False):
        native, path = encode_latent(encoder, context.values.to(device), source_domain, strategy="full",
                                     block_size=(1, 1, 1), halo=(0, 0, 0), precision="float32")
    if path != "full" or native.ndim != 5 or native.shape[2:] != core.shape or not torch.isfinite(native).all():
        raise ValueError("Expected a finite, full-volume native posterior on the valid-core grid.")
    start = stats.normalize(native, mask)
    if torch.count_nonzero(start.masked_select(~mask)).item() or not bool(torch.isfinite(start).all()):
        raise ValueError("Expected a finite translator input and an identically zero normalized exterior.")
    start_hash = storage_tensor_sha256(start)


    return PreparedSource(source_domain, expected_shape, source, context, foreground, core, mask, native, start, start_hash)


@torch.inference_mode()
def score_direction(*, prepared, decoder, translator, artifact, stats, target, target_domain, wrong_domain,
                    raw_stage1_source, calibrated_stage1_source, calibrate, case_seed, device, mark=print):
    """Keep the decoder and renderer fixed across the three native-exterior transport arms."""
    device = torch.device(device)
    _require_linear_decoder_output_activation(decoder)
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (decoder, translator)):
        raise ValueError("Frozen decoder and translator required.")
    source_domain, expected_shape, source, context = (
        prepared.source_domain, prepared.expected_shape, prepared.source, prepared.context
    )
    if len({source_domain, target_domain, wrong_domain}) != 3:
        raise ValueError("Expected distinct source, target and wrong domains.")
    if len({d.contrast for d in (source_domain, target_domain, wrong_domain)}) != 1:
        raise ValueError("All domain labels must use the same contrast.")
    foreground, core, mask = prepared.foreground, prepared.core, prepared.mask
    native, start, start_hash = prepared.native, prepared.start, prepared.start_hash
    predictions = {"photometry_only": artifact.render_target(context, target_domain).float().cpu()}
    traces = {}
    for name, request in (("zero", None), ("correct", target_domain), ("wrong", wrong_domain)):
        mark(f"{name}: fixed {target_domain.label} decoder and renderer")
        endpoint, times, labels = ((start, [], []) if request is None else
                                   _solve(translator, start, source_domain, request, device, case_seed))
        if start_hash != storage_tensor_sha256(start):
            raise ValueError("Transport modified the shared start.")
        if endpoint.shape != start.shape or not bool(torch.isfinite(endpoint).all()):
            raise ValueError("Nonfinite or malformed endpoint.")
        decoder_input = decoder_input_latent(
            endpoint, stats=stats, native_source=native, valid_core=mask, policy=SOURCE_NATIVE_EXTERIOR,
        )
        with _matched_rng(device, case_seed), torch.autocast(device.type, dtype=torch.bfloat16):
            batch = decoder.decode(decoder_input, [target_domain])
        raw = batch.detach().float().cpu()
        bounded = adapt_decoder_evaluation_range(raw, foreground, expected_shape=tuple(source.shape),
                                                role="primary_unified_output")
        prediction = artifact.render_target(context.with_values(bounded.bounded), target_domain).float().cpu()
        if (prediction.shape != source.shape or not bool(torch.isfinite(prediction).all())
                or bool(torch.count_nonzero(prediction[~foreground]))):
            raise ValueError("Fixed rendering must preserve the source grid and zero background.")
        predictions[name] = prediction
        traces[name] = {"velocity_evaluations": len(times), "velocity_times": times, "velocity_labels": labels,
                        "neural_request": request.label if request else None,
                        "decoder_domain": target_domain.label, "renderer_domain": target_domain.label,
                        "normalized_endpoint_sha256": storage_tensor_sha256(endpoint),
                        "decoder_input_sha256": storage_tensor_sha256(decoder_input)}
        del prediction, batch, raw, bounded, decoder_input, endpoint

    mark("score six fixed methods versus the acquired target")
    target_layout = adapt_full_volume_layout(target, role="measured target", expected_shape=expected_shape)
    target = validated_source_volume(target_layout.model_batch(), target_domain.label)
    baseline = adapt_full_volume_layout(calibrated_stage1_source, role="calibrated Stage 1 source",
                                        expected_shape=expected_shape).model_batch().detach().float().cpu()
    stage1 = adapt_full_volume_layout(raw_stage1_source, role="archived Stage 1 source",
                                      expected_shape=expected_shape).model_batch().detach().float().cpu()
    target_foreground = target != 0
    shared = foreground & target_foreground
    if not bool(shared.any()):
        raise ValueError("Source and target have no shared foreground.")
    target_array, shared_array = target[0, 0].numpy(), shared[0, 0].numpy()
    score_regions = {"shared_foreground": shared_array}

    def score(prediction):
        pred_array = prediction[0, 0].numpy()
        diff = _differences(prediction, target, score_regions)["shared_foreground"]
        return {
            "full_nrmse": official_task3_nrmse(pred_array, target_array),
            "shared_foreground_nrmse": official_task3_nrmse(pred_array, target_array, mask=shared_array),
            "shared_foreground_rmse": diff["rmse"],
            "rendered_prediction_sha256": storage_tensor_sha256(prediction),
        }

    trace = {
        "requests": traces, "case_seed": case_seed, "encoder_calls_scientific_source": 1,
        "integration_count": 2, "masked_start_sha256": start_hash,
        "native_posterior_sha256": storage_tensor_sha256(native),
        "core_mask_sha256": storage_tensor_sha256(core), "latent_core_count": int(core.sum()),
        "latent_grid_count": core.numel(), "source_foreground_count": int(foreground.sum()),
        "target_foreground_count": int(target_foreground.sum()), "shared_foreground_count": int(shared.sum()),
        "source_shape": list(expected_shape), "spatial_shape": list(target_array.shape),
    }
    predictions.update(acquired_source=source, stage1_source=stage1)
    raw_scores = {name: score(value)
                  for name, value in predictions.items()}
    calibrated_scores = {}
    learned_calibrated = {}
    for name, value in predictions.items():
        mark(f"matched per-prediction calibration and scoring: {name}")
        if name == "stage1_source":
            # The caller already applied the same calibrator to this source reconstruction
            # and authenticated the result against the archived calibrated identity.
            corrected = baseline
        else:
            transformed = calibrate(value.reshape(expected_shape))
            corrected = adapt_full_volume_layout(
                transformed, role=f"calibrated {name}", expected_shape=expected_shape
            ).model_batch().detach().float().cpu()
        if torch.count_nonzero(corrected[~foreground]).item():
            raise ValueError(f"Calibrated {name} has nonzero signal outside the fixed source mask.")
        if storage_tensor_sha256(value) != raw_scores[name]["rendered_prediction_sha256"]:
            raise ValueError(f"Calibration modified the original {name} prediction in place.")
        calibrated_scores[name] = score(corrected)
        if name in ("correct", "wrong"):
            learned_calibrated[name] = corrected
        del corrected
    def signed_deltas(scores):
        return {f"correct_minus_{name}": scores["correct"]["full_nrmse"] - result["full_nrmse"]
                for name, result in scores.items() if name != "correct"}
    regions = {"source_foreground": foreground[0, 0].numpy(), **score_regions}
    return {
        "contract": "P0006_all_directions_native_exterior_matched_calibration_v1",
        "source_domain": source_domain.label, "target_domain": target_domain.label,
        "wrong_neural_domain": wrong_domain.label,
        "raw_scores": raw_scores, "calibrated_scores": calibrated_scores,
        "raw_full_nrmse_deltas": signed_deltas(raw_scores),
        "calibrated_full_nrmse_deltas": signed_deltas(calibrated_scores),
        "direct_correct_vs_wrong": {
            "before": _differences(predictions["correct"], predictions["wrong"], regions),
            "after": _differences(learned_calibrated["correct"], learned_calibrated["wrong"], regions),
        },
        "trace": trace,
    }
