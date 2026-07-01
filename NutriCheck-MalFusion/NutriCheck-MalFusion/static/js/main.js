/* NutriCheck — prediction form logic */
(function () {
  var form = document.getElementById("assess-form");
  if (!form) return;

  var btn        = document.getElementById("assess-btn");
  var errBox     = document.getElementById("form-error");
  var emptyState = document.getElementById("result-empty");
  var bodyState  = document.getElementById("result-body");
  var banner     = document.getElementById("result-banner");
  var arc        = document.getElementById("gauge-arc");
  var pctEl      = document.getElementById("gauge-pct");
  var labelEl    = document.getElementById("status-label");
  var confEl     = document.getElementById("conf-text");
  var blurbEl    = document.getElementById("result-blurb");
  var probList   = document.getElementById("prob-list");

  var CIRC = 314; // 2*pi*50

  /* Pill toggle behaviour */
  form.addEventListener("click", function (ev) {
    var pill = ev.target.closest(".pill");
    if (!pill) return;
    var group = pill.closest(".pills");
    var pills = group.querySelectorAll(".pill");
    for (var i = 0; i < pills.length; i++) pills[i].setAttribute("aria-pressed", "false");
    pill.setAttribute("aria-pressed", "true");
    group.setAttribute("data-value", pill.getAttribute("data-val"));
  });

  /* Gather every field into a {name: value} object */
  function collect() {
    var data = {};
    form.querySelectorAll("input[data-name]").forEach(function (el) {
      data[el.getAttribute("data-name")] = el.value;
    });
    form.querySelectorAll("select[data-name]").forEach(function (el) {
      data[el.getAttribute("data-name")] = el.value;
    });
    form.querySelectorAll(".pills[data-name]").forEach(function (el) {
      data[el.getAttribute("data-name")] = el.getAttribute("data-value");
    });
    return data;
  }

  function showError(msg) {
    errBox.textContent = msg;
    errBox.classList.add("show");
    errBox.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function clearError() { errBox.classList.remove("show"); }

  function setLoading(on) {
    btn.disabled = on;
    btn.innerHTML = on
      ? '<span class="spin"></span><span>Assessing…</span>'
      : '<svg><use href="#i-spark"/></svg><span>Assess nutrition status</span>';
  }

  function render(res) {
    // banner tone
    banner.className = "result-banner tone-" + (res.tone || "watch");
    labelEl.textContent = res.display_label || res.label;
    confEl.textContent = res.confidence + "% confidence";
    blurbEl.textContent = res.blurb || "";

    // gauge animation
    var conf = Math.max(0, Math.min(100, res.confidence));
    pctEl.textContent = Math.round(conf) + "%";
    arc.style.strokeDashoffset = CIRC;
    requestAnimationFrame(function () {
      arc.style.strokeDashoffset = CIRC - (CIRC * conf / 100);
    });

    // probability bars (sorted high → low)
    var probs = res.probabilities || {};
    var rows = Object.keys(probs).map(function (k) { return [k, probs[k]]; });
    rows.sort(function (a, b) { return b[1] - a[1]; });
    probList.innerHTML = "";
    rows.forEach(function (r) {
      var div = document.createElement("div");
      div.className = "prob-row";
      div.innerHTML =
        '<div class="top"><span>' + r[0] + '</span><span>' + r[1] + '%</span></div>' +
        '<div class="bar"><div class="fill" style="width:0%"></div></div>';
      probList.appendChild(div);
      var fill = div.querySelector(".fill");
      requestAnimationFrame(function () { fill.style.width = r[1] + "%"; });
    });

    emptyState.style.display = "none";
    bodyState.classList.add("show");
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    clearError();
    setLoading(true);

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (out) {
        setLoading(false);
        if (!out.ok) { showError(out.j.error || "Something went wrong."); return; }
        render(out.j);
      })
      .catch(function () {
        setLoading(false);
        showError("Could not reach the server. Is the app running?");
      });
  });

  form.addEventListener("reset", function () {
    clearError();
    bodyState.classList.remove("show");
    emptyState.style.display = "";
    // reset pills to first option
    form.querySelectorAll(".pills").forEach(function (g) {
      var pills = g.querySelectorAll(".pill");
      pills.forEach(function (p, i) { p.setAttribute("aria-pressed", i === 0 ? "true" : "false"); });
      if (pills[0]) g.setAttribute("data-value", pills[0].getAttribute("data-val"));
    });
  });
})();
