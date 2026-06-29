#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted);

// FFI-safe, host/device-flattened aggregate of llama.cpp's per-device memory
// breakdown. The native breakdown is a C++ `std::map` keyed by backend buffer
// type and so cannot cross FFI; this collapses it into a host side and a device
// side. Every field is in bytes. Device fields aggregate every non-host backend
// buffer type (the GPU backends); host fields aggregate the host (CPU) backend.
// The `*_context` fields are the KV-cache plus any recurrent-state memory — the
// host/device split of `context` is the load-bearing KV-placement signal.
struct llama_rs_memory_breakdown {
    uint64_t device_model;
    uint64_t device_context;
    uint64_t device_compute;
    uint64_t host_model;
    uint64_t host_context;
    uint64_t host_compute;
};

// Read the constructed context's per-device memory breakdown, flattening
// llama.cpp's `std::map` into the host/device aggregate above. A null context
// yields an all-zero breakdown.
struct llama_rs_memory_breakdown llama_rs_memory_breakdown_data(const struct llama_context * ctx);

// Set the verbosity threshold of llama.cpp's `common` logger (a separate logger
// from `llama_log_set`/`ggml_log_set`, which `send_logs_to_tracing` hooks). A
// message logged via LOG_INF/LOG_WRN/etc. is emitted only when its level
// (DEBUG=5, TRACE=4, INFO=3, WARN=2, ERROR=1) is <= `verbosity`. The `common`
// logger writes straight to stdout/stderr and is not routed to tracing, so its
// output corrupts a TUI; lowering the threshold (e.g. to 2 to keep WARN/ERROR
// but drop INFO) silences chatty subsystems such as the speculative driver.
void llama_rs_common_log_set_verbosity_thold(int verbosity);

void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif
