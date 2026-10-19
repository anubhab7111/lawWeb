// Live capture of the real LawWeb chat page at 2x: a screenshot loop (screencast drops content under CSS zoom),
// a DOM timeline for status labels / first text / completion, and nvidia-smi logs on the same clock.
// Usage: node capture2.mjs <outDir> <question> [maxSeconds]   (SCROLL_TO_QUESTION=1 for translated answers)
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const puppeteer = (await import(
  "/home/ushtro/.npm/_npx/5ceb456240d0f2bf/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js"
)).default;
const CHROME =
  "/home/ushtro/.cache/hyperframes/chrome/chrome-headless-shell/linux-152.0.7977.30/chrome-headless-shell-linux64/chrome-headless-shell";

const [outDir, question, maxSecondsArg] = process.argv.slice(2);
const maxSeconds = Number(maxSecondsArg || 240);
const framesDir = path.join(outDir, "frames");
fs.mkdirSync(framesDir, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString(), ...a);
const events = [];
const mark = (type, extra = {}) => events.push({ type, t: Date.now(), ...extra });

const smiGpu = fs.createWriteStream(path.join(outDir, "smi_gpu.csv"));
const smiApps = fs.createWriteStream(path.join(outDir, "smi_apps.csv"));
const pipeStamped = (child, stream) => {
  let buf = "";
  child.stdout.on("data", (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) stream.write(`${Date.now()},${line}\n`);
    }
  });
};
const gpu = spawn("nvidia-smi", [
  "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
  "--format=csv,noheader,nounits", "-lms", "250",
]);
const apps = spawn("nvidia-smi", [
  "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits", "-lms", "500",
]);
pipeStamped(gpu, smiGpu);
pipeStamped(apps, smiApps);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox", "--hide-scrollbars", "--font-render-hinting=none"],
  defaultViewport: { width: 1280, height: 720, deviceScaleFactor: 2 },
});
const page = await browser.newPage();
await page.goto("http://localhost:5173/#/chat", { waitUntil: "networkidle0", timeout: 60000 });
await page.waitForSelector("textarea", { timeout: 30000 });
await sleep(800);

const frames = [];
let capturing = true;
const loop = (async () => {
  while (capturing) {
    const t = Date.now();
    const file = `f${String(frames.length).padStart(5, "0")}.jpg`;
    await page.screenshot({ path: path.join(framesDir, file), type: "jpeg", quality: 88 });
    frames.push({ file, t });
  }
})();
mark("capture_start");

let done = false;
page.on("requestfinished", (req) => {
  if (req.url().includes("/chat/stream") && req.method() === "POST") { mark("stream_finished"); done = true; }
});

await sleep(800);
await page.click("textarea");
mark("typing_start");
await page.type("textarea", question, { delay: 28 });
mark("typing_end");
await sleep(350);
await page.keyboard.press("Enter");
mark("submit");
log("submitted");

const timeline = [];
const t0 = Date.now();
let firstText = null;
while (!done && Date.now() - t0 < maxSeconds * 1000) {
  const snap = await page.evaluate(() => {
    const blocks = document.querySelectorAll(".fade-up");
    const last = blocks[blocks.length - 1];
    if (!last) return null;
    const prose = last.querySelector(".prose");
    return { all: last.innerText.slice(0, 300), proseLen: prose ? prose.innerText.length : 0 };
  });
  if (snap) {
    timeline.push({ t: Date.now(), ...snap });
    if (firstText === null && snap.proseLen > 0) { firstText = Date.now(); mark("first_text"); log("first text"); }
  }
  await sleep(150);
}
mark(done ? "done" : "timeout");
log(done ? "done" : "TIMEOUT");
await sleep(1500);
if (process.env.SCROLL_TO_QUESTION) {
  await page.evaluate(() => {
    const blocks = document.querySelectorAll(".fade-up");
    const last = blocks[blocks.length - 1];
    (last.previousElementSibling || last).scrollIntoView({ behavior: "smooth", block: "start" });
  });
  mark("scrolled_to_question");
  await sleep(3500);
}
await sleep(1000);
capturing = false;
await loop;
mark("capture_end");
const finalText = await page.evaluate(() => {
  const blocks = document.querySelectorAll(".fade-up");
  return blocks.length ? blocks[blocks.length - 1].innerText : "";
});
gpu.kill(); apps.kill();
await browser.close();
fs.writeFileSync(path.join(outDir, "capture.json"), JSON.stringify({ question, events, frames, timeline, finalText }, null, 1));
log(`frames=${frames.length} events=${events.map((e) => e.type).join(",")}`);
