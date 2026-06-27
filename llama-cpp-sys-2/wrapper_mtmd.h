#include "llama.cpp/tools/mtmd/mtmd.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"

#ifdef __cplusplus
extern "C" {
#endif

// Redirect clip.cpp's internal logger to `callback`. clip.cpp (the vision projector / CLIP encoder)
// and mtmd.cpp use a logger separate from `llama_log_set` / `ggml_log_set` — the `g_logger_state`
// global in clip-impl.h — whose default callback writes straight to stderr, corrupting a TUI. clip
// exposes no public setter, so this shim assigns `g_logger_state` directly. Routing it to the same
// `ggml_log_callback` that `send_logs_to_tracing` installs for llama/ggml sends projector and image
// logs to tracing instead of the terminal. `user_data` is forwarded to the callback unchanged.
void llama_rs_clip_log_set(ggml_log_callback callback, void * user_data);

#ifdef __cplusplus
}
#endif
