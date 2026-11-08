// 在 Windows release 版本中隐藏控制台窗口
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    colab_pilot_lib::run()
}
