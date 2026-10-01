// === 1) หา element ที่ต้องใช้ ===
const form = document.getElementById("predict-form");
const button = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const errorBox = document.getElementById("error");

// === 2) เมื่อกดปุ่ม Predict ===
form.addEventListener("submit", async (event) => {
  event.preventDefault();                     // ไม่ให้เบราว์เซอร์รีโหลดหน้า

  // รวบรวมค่าจากฟอร์มเป็น object เช่น { age: "67", hypertension: "0", ... }
  const data = Object.fromEntries(new FormData(form).entries());

  button.disabled = true;
  button.textContent = "กำลังประเมิน...";
  errorBox.classList.add("hidden");

  try {
    // ส่งข้อมูลไปให้ Flask ที่ /predict
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "เกิดข้อผิดพลาด");
    }
    showResult(result);
  } catch (err) {
    resultContent.classList.add("hidden");
    resultEmpty.classList.add("hidden");
    resultBox.className = "card result";
    errorBox.textContent = "⚠️ " + err.message;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
});

// === 3) แสดงผลลัพธ์และความหมาย ===
function showResult(result) {
  const score = result.score;
  const threshold = result.threshold;
  const isRisk = result.prediction === 1;

  document.getElementById("result-label").textContent =
    isRisk ? "มีความเสี่ยง ควรพบแพทย์" : "ความเสี่ยงต่ำ";
  document.getElementById("result-score").textContent = score.toFixed(2);
  document.getElementById("result-threshold").textContent = threshold.toFixed(2);

  // แถบคะแนน: ความยาว = คะแนน, เส้นดำ = เกณฑ์ตัดสิน
  document.getElementById("bar-fill").style.width = (score * 100) + "%";
  document.getElementById("bar-marker").style.left = (threshold * 100) + "%";

  document.getElementById("result-meaning").textContent = isRisk
    ? `คะแนน ${score.toFixed(2)} สูงกว่าเกณฑ์ ${threshold.toFixed(2)} แปลว่าข้อมูลของคุณคล้ายกับกลุ่มผู้ที่เคยเป็นโรคหลอดเลือดสมองในข้อมูลที่ใช้สอนโมเดล คะแนนนี้ไม่ใช่เปอร์เซ็นต์โอกาสเป็นโรค`
    : `คะแนน ${score.toFixed(2)} ต่ำกว่าเกณฑ์ ${threshold.toFixed(2)} แปลว่าข้อมูลของคุณคล้ายกับกลุ่มผู้ที่ไม่เป็นโรคในข้อมูลที่ใช้สอนโมเดล`;

  document.getElementById("result-advice").textContent = isRisk
    ? "ผลนี้ไม่ใช่การวินิจฉัย แต่แนะนำให้ตรวจสุขภาพ วัดความดัน และตรวจน้ำตาลในเลือดกับแพทย์ (โมเดลตั้งใจให้เตือนไว้ก่อน จึงอาจเตือนเกินจริงได้)"
    : "ยังควรดูแลสุขภาพต่อเนื่อง เช่น ควบคุมความดัน น้ำตาล และน้ำหนัก เพราะโมเดลอาจพลาดได้ประมาณ 1 ใน 5 ของผู้ป่วยจริง";

  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultBox.className = "card result " + (isRisk ? "high" : "low");
}
