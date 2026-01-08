use super::{BenchmarkHost, Measurement};

/// Implementation for WebAssembly environments (Browsers, Node.js).
/// Uses the `performance` API for timing.
pub struct Platform;

impl BenchmarkHost for Platform {
    type TimePoint = f64; // Time in milliseconds

    fn now() -> Self::TimePoint {
        use wasm_bindgen::JsCast;
        use wasm_bindgen::JsValue;

        // 1. Try to get the window object (Browser)
        if let Some(win) = web_sys::window() {
            if let Some(perf) = win.performance() {
                return perf.now();
            }
        }

        // 2. Try to get the global object (Node.js / Web Workers)
        let global = js_sys::global();
        if let Ok(perf_val) = js_sys::Reflect::get(&global, &JsValue::from_str("performance")) {
            if perf_val.is_object() {
                let perf: web_sys::Performance = perf_val.unchecked_into();
                return perf.now();
            }
        }

        // 3. Fallback to Date (Less precise)
        js_sys::Date::now()
    }

    fn measure(start: Self::TimePoint) -> Measurement {
        let current = Self::now();
        Measurement {
            ticks: None,
            micros: (current - start) * 1000.0, // ms to us
        }
    }

    fn print(s: &str) {
        use wasm_bindgen::JsValue;
        use web_sys::console;
        // Logs to the browser console or Node.js stdout
        console::log_1(&JsValue::from_str(s));
    }

    fn platform_name() -> &'static str {
        "Wasm32"
    }
}
