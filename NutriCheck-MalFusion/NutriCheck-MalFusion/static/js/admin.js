/* NutriCheck — admin panel logic */
(function () {
  var lockBox  = document.getElementById("lock-box");
  var content  = document.getElementById("admin-content");

  function unlock() {
    lockBox.style.display = "none";
    content.style.display = "block";
    loadStatus();
  }

  /* ---------- Login ---------- */
  var loginBtn = document.getElementById("login-btn");
  var passInput = document.getElementById("passcode");

  function doLogin() {
    var code = passInput.value;
    loginBtn.disabled = true;
    fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passcode: code }),
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (o) {
        loginBtn.disabled = false;
        if (o.ok && o.j.ok) { unlock(); window.showToast("Welcome, admin"); }
        else { window.showToast(o.j.error || "Incorrect passcode"); passInput.focus(); }
      })
      .catch(function () { loginBtn.disabled = false; window.showToast("Server error"); });
  }
  if (loginBtn) loginBtn.addEventListener("click", doLogin);
  if (passInput) passInput.addEventListener("keydown", function (e) { if (e.key === "Enter") doLogin(); });

  if (window.__UNLOCKED__) unlock();

  /* ---------- Status ---------- */
  function fmt(v) { return (v === null || v === undefined) ? "—" : v; }
  function pct(v) { return (v === null || v === undefined) ? "—" : (Math.round(v * 1000) / 10) + "%"; }

  function loadStatus() {
    fetch("/api/admin/status")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var badge = document.getElementById("model-badge");
        if (d.model_loaded) { badge.className = "badge badge-on"; badge.textContent = "● Model ready"; }
        else { badge.className = "badge badge-off"; badge.textContent = "● Not trained"; }

        var s = d.status || {};
        var m = s.metrics || {};
        document.getElementById("st-acc").textContent  = pct(m.accuracy);
        document.getElementById("st-f1").textContent   = pct(m.f1);
        document.getElementById("st-prec").textContent = pct(m.precision);
        document.getElementById("st-rec").textContent  = pct(m.recall);

        var data = s.data || {};
        document.getElementById("st-rows").textContent = fmt(data.total_rows);
        document.getElementById("st-new").textContent  = fmt(d.new_records);
        document.getElementById("st-feat").textContent = fmt(m.n_features);
        document.getElementById("st-when").textContent = fmt(m.trained_at) === "—" ? "—" : m.trained_at;

        // components
        var box = document.getElementById("st-components");
        box.innerHTML = "";
        var comp = m.components || {};
        var avail = d.components_available || {};
        var pairs = [
          ["Random Forest ×2", comp.random_forest],
          ["SVM ×2", comp.svm],
          ["XGBoost ×1", comp.xgboost],
          ["CatBoost ×1", comp.catboost],
        ];
        pairs.forEach(function (p) {
          var el = document.createElement("span");
          el.className = "comp";
          el.textContent = p[0] + (p[1] ? " · " + p[1] : "");
          box.appendChild(el);
        });
        Object.keys(avail).forEach(function (k) {
          var el = document.createElement("span");
          el.className = "comp";
          el.textContent = (avail[k] ? "✓ " : "✗ ") + k;
          el.style.color = avail[k] ? "var(--good)" : "var(--muted)";
          box.appendChild(el);
        });
      })
      .catch(function () {});
  }

  /* ---------- Render health report ---------- */
  function renderReport(report) {
    var wrap = document.getElementById("train-result");
    var issues = document.getElementById("issues-list");
    var fixes = document.getElementById("fixes-list");
    issues.innerHTML = "";
    fixes.innerHTML = "";

    (report.issues || []).forEach(function (txt) {
      var li = document.createElement("li");
      li.className = "issue-item";
      li.textContent = txt;
      issues.appendChild(li);
    });
    (report.fixes || []).forEach(function (fx) {
      var li = document.createElement("li");
      li.className = "fix-item";
      li.innerHTML = '<div class="p">' + fx.problem + '</div><div class="a">→ ' + fx.applied + '</div>';
      fixes.appendChild(li);
    });
    if (!report.fixes || report.fixes.length === 0) {
      var li = document.createElement("li");
      li.className = "fix-item";
      li.textContent = "No fixes needed — data was already clean.";
      fixes.appendChild(li);
    }
    wrap.style.display = "block";
  }

  /* ---------- Check data only ---------- */
  var checkBtn = document.getElementById("check-btn");
  if (checkBtn) checkBtn.addEventListener("click", function () {
    checkBtn.disabled = true;
    var original = checkBtn.innerHTML;
    checkBtn.innerHTML = '<span class="spin" style="border-color:rgba(0,0,0,.2);border-top-color:var(--primary)"></span><span>Checking…</span>';
    fetch("/api/admin/health")
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (o) {
        checkBtn.disabled = false; checkBtn.innerHTML = original;
        if (!o.ok) { window.showToast(o.j.error || "Check failed"); return; }
        renderReport(o.j.report);
        window.showToast("Checked " + o.j.rows_before + " rows");
      })
      .catch(function () { checkBtn.disabled = false; checkBtn.innerHTML = original; window.showToast("Server error"); });
  });

  /* ---------- Train ---------- */
  var trainBtn = document.getElementById("train-btn");
  if (trainBtn) trainBtn.addEventListener("click", function () {
    trainBtn.disabled = true;
    var original = trainBtn.innerHTML;
    trainBtn.innerHTML = '<span class="spin"></span><span>Training…</span>';
    fetch("/api/admin/train", { method: "POST" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (o) {
        trainBtn.disabled = false; trainBtn.innerHTML = original;
        if (!o.ok) { window.showToast(o.j.error || "Training failed"); return; }
        renderReport(o.j.status.health);
        loadStatus();
        var acc = o.j.status.metrics.accuracy;
        window.showToast("Trained · accuracy " + (acc !== null ? Math.round(acc * 1000) / 10 + "%" : "n/a"));
      })
      .catch(function () { trainBtn.disabled = false; trainBtn.innerHTML = original; window.showToast("Server error"); });
  });

  /* ---------- Save record ---------- */
  var saveBtn = document.getElementById("save-rec-btn");
  var recError = document.getElementById("rec-error");
  if (saveBtn) saveBtn.addEventListener("click", function () {
    recError.classList.remove("show");
    var data = {};
    document.querySelectorAll("[data-rec]").forEach(function (el) {
      data[el.getAttribute("data-rec")] = el.value;
    });
    saveBtn.disabled = true;
    fetch("/api/admin/save-record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (o) {
        saveBtn.disabled = false;
        if (!o.ok) {
          recError.textContent = o.j.error || "Could not save.";
          recError.classList.add("show");
          return;
        }
        window.showToast("Saved · " + o.j.new_records + " new record(s)");
        loadStatus();
      })
      .catch(function () { saveBtn.disabled = false; window.showToast("Server error"); });
  });

  /* ---------- Logout ---------- */
  var logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) logoutBtn.addEventListener("click", function () {
    fetch("/api/admin/logout", { method: "POST" }).then(function () {
      content.style.display = "none";
      lockBox.style.display = "block";
      window.showToast("Logged out");
    });
  });
})();
