#include "wrapper_mtmd.h"

// clip-impl.h declares `extern struct clip_logger_state g_logger_state;` (defined in clip.cpp) plus
// the `clip_logger_state` layout. Including it here ties this shim to clip's actual declaration rather
// than re-declaring the struct, so a layout change upstream is a compile error here, not silent
// memory corruption.
#include "llama.cpp/tools/mtmd/clip-impl.h"

// Assign clip's internal logger so projector / image-encode logs route to `callback` instead of the
// default stderr writer. clip has no public setter (clip-impl.h is internal), which is why this lives
// in a wrapper TU compiled only under the `mtmd` feature.
extern "C" void llama_rs_clip_log_set(ggml_log_callback callback, void * user_data) {
    g_logger_state.log_callback = callback;
    g_logger_state.log_callback_user_data = user_data;
}
