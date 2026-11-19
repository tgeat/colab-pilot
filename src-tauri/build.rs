fn main() {
    println!("cargo:rerun-if-changed=remote/telemetry_daemon.py");
    println!("cargo:rerun-if-changed=../train.py");
    println!(
        "cargo:rerun-if-changed=../.agents/skills/colab-llm-bluegreen/scripts/llm_local_gateway.py"
    );
    println!(
        "cargo:rerun-if-changed=../.agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py"
    );
    println!(
        "cargo:rerun-if-changed=../.agents/skills/colab-llm-bluegreen/scripts/llm_metrics_sidecar.py"
    );
    println!("cargo:rerun-if-changed=../.agents/skills/colab-dl-relay/scripts/dl_relay_runner.py");
    println!(
        "cargo:rerun-if-changed=../.agents/skills/colab-dl-relay/scripts/colab_relay_adapter.py"
    );

    // 开发/检查态（非 release 打包）在尚未执行 `npm run build` 生成 `../dist` 时，
    // 自动将 frontendDist 指向构建缓存目录中的占位目录，使 `cargo check --all-targets --features custom-protocol` 开箱通过；
    // 正式 `release` 打包或 `../dist` 已存在时严格使用真实前端产物目录。
    let dist_exists = std::path::Path::new("../dist").exists();
    let is_release = std::env::var("PROFILE").as_deref() == Ok("release");
    if !dist_exists && !is_release && std::env::var_os("TAURI_CONFIG").is_none() {
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let placeholder = std::path::PathBuf::from(out_dir).join("placeholder_dist");
            let _ = std::fs::create_dir_all(&placeholder);
            let norm = placeholder.to_string_lossy().replace('\\', "/");
            let cfg_json = format!(r#"{{"build":{{"frontendDist":"{norm}"}}}}"#);
            std::env::set_var("TAURI_CONFIG", &cfg_json);
            println!("cargo:rustc-env=TAURI_CONFIG={cfg_json}");
        }
    }

    tauri_build::build()
}
