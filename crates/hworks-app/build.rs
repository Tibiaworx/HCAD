// Embed the HCAD icon into hcad.exe on Windows so Explorer, the taskbar, and shortcuts show the
// logo. Non-fatal: if the resource compiler (rc.exe / llvm-rc) isn't available, the build still
// succeeds — the runtime `set_window_icon` call covers the taskbar regardless.
fn main() {
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=assets/hcad.ico");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/hcad.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=HCAD icon not embedded ({e}); taskbar icon still set at runtime");
        }
    }
}
