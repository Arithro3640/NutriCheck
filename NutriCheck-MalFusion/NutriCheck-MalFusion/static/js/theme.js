/* Theme switcher — applies a theme to <html data-theme> and remembers it. */
(function () {
  var KEY = "nutricheck-theme";
  var root = document.documentElement;

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    var btns = document.querySelectorAll(".swatch");
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute("aria-pressed",
        btns[i].getAttribute("data-theme") === theme ? "true" : "false");
    }
  }

  // Restore saved theme on load.
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  if (saved) apply(saved);

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest(".swatch");
    if (!btn) return;
    var theme = btn.getAttribute("data-theme");
    apply(theme);
    try { localStorage.setItem(KEY, theme); } catch (e) {}
  });

  // Expose a small toast helper used by other scripts.
  window.showToast = function (msg) {
    var t = document.getElementById("toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(window.__toastT);
    window.__toastT = setTimeout(function () { t.classList.remove("show"); }, 2600);
  };
})();
