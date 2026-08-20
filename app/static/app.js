(function () {
  const form = document.getElementById("estimate-form");
  const brandSelect = document.getElementById("brand");
  const conditionSelect = document.getElementById("condition");
  const promptInput = document.getElementById("prompt");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const midPriceEl = document.getElementById("mid-price");
  const rangePriceEl = document.getElementById("range-price");
  const confidenceEl = document.getElementById("confidence");
  const rationaleEl = document.getElementById("rationale");
  const compsList = document.getElementById("comps-list");

  function money(value) {
    return Number(value).toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    });
  }

  function setStatus(message, isError) {
    statusEl.hidden = !message;
    statusEl.textContent = message || "";
    statusEl.className = isError ? "status error" : "status";
  }

  function renderResult(payload) {
    const estimate = payload.estimate;
    midPriceEl.textContent = money(estimate.mid);
    rangePriceEl.textContent = money(estimate.low) + " – " + money(estimate.high);
    confidenceEl.textContent = "Confidence: " + estimate.confidence;
    rationaleEl.textContent = estimate.rationale;

    compsList.innerHTML = "";
    (payload.comparables || []).forEach(function (hit) {
      const item = document.createElement("li");
      const title = document.createElement("div");
      title.className = "comp-title";
      title.innerHTML =
        "<span></span><span></span>";
      title.children[0].textContent =
        hit.year + " " + hit.brand + " " + hit.model;
      title.children[1].textContent = money(hit.price);

      const meta = document.createElement("p");
      meta.className = "comp-meta";
      meta.textContent = [
        hit.condition,
        hit.mileage != null ? Number(hit.mileage).toLocaleString() + " miles" : null,
        hit.horsepower != null ? hit.horsepower + " hp" : null,
        hit.torque_lbft != null ? hit.torque_lbft + " lb-ft" : null,
      ]
        .filter(Boolean)
        .join(" · ");

      item.appendChild(title);
      item.appendChild(meta);
      compsList.appendChild(item);
    });

    resultEl.hidden = false;
  }

  async function loadOptions() {
    const response = await fetch("/api/options");
    if (!response.ok) {
      throw new Error("Could not load brand options.");
    }
    const data = await response.json();
    brandSelect.innerHTML = '<option value="">Select brand</option>';
    data.brands.forEach(function (brand) {
      const option = document.createElement("option");
      option.value = brand;
      option.textContent = brand;
      brandSelect.appendChild(option);
    });
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    resultEl.hidden = true;
    setStatus("Searching comparables and estimating…", false);
    submitBtn.disabled = true;

    try {
      const response = await fetch("/api/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brand: brandSelect.value,
          condition: conditionSelect.value,
          prompt: promptInput.value.trim(),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail;
        const message =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map(function (item) { return item.msg || JSON.stringify(item); }).join(" ")
              : "Estimate failed.";
        throw new Error(message);
      }
      setStatus("", false);
      renderResult(payload);
    } catch (error) {
      setStatus(error.message || "Estimate failed.", true);
    } finally {
      submitBtn.disabled = false;
    }
  });

  loadOptions().catch(function (error) {
    brandSelect.innerHTML = '<option value="">Unavailable</option>';
    setStatus(error.message, true);
  });
})();
