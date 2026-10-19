import { test } from "node:test";
import assert from "node:assert/strict";
import { depthTracker, parseLine } from "../src/components/listLine.ts";

function layout(text: string) {
  const depthOf = depthTracker();
  return text.split("\n").map((raw) => {
    const line = parseLine(raw);
    return { depth: depthOf(line), label: line.marker?.label ?? null, text: line.text };
  });
}

// Verbatim from a qwen3:4b-instruct answer to "Can the police arrest me without a warrant?" (trailing
// double spaces included); before this fix the (i)–(v) lines rendered flush-left with no marker.
const ARREST = `- **Key provisions:**
  - **CrPC § 41** and **BNSS § 35**: Allow arrest without warrant if (a) the person commits a cognizable offence in the presence of a police officer, or (b) there is reasonable suspicion, and the officer is satisfied that arrest is necessary to:
    (i) prevent further offence;
    (ii) ensure proper investigation;
    (iii) prevent destruction or tampering with evidence;
  - **CrPC § 151**: Enables arrest to prevent commission of a cognizable offence.

- **Mandatory conditions:**
  - The arrestee must be informed **immediately** of the grounds of arrest (CrPC § 50).

This is general legal information.`;

test("roman enumerations nest under the bullet that introduces them", () => {
  const rows = layout(ARREST);
  assert.deepEqual(rows[0], { depth: 0, label: "•", text: "**Key provisions:**" });
  assert.equal(rows[1].depth, 1);
  assert.equal(rows[1].label, "•");
  assert.deepEqual(rows[2], { depth: 2, label: "(i)", text: "prevent further offence;" });
  assert.deepEqual(rows[3], { depth: 2, label: "(ii)", text: "ensure proper investigation;" });
  assert.deepEqual(rows[4], { depth: 2, label: "(iii)", text: "prevent destruction or tampering with evidence;" });
  assert.equal(rows[5].depth, 1);
  assert.equal(rows[5].label, "•");
});

test("a blank line keeps the list open and an unindented paragraph closes it", () => {
  const rows = layout(ARREST);
  assert.deepEqual([rows[7].depth, rows[7].label], [0, "•"]);
  assert.deepEqual([rows[8].depth, rows[8].label], [1, "•"]);
  assert.deepEqual(rows[10], { depth: 0, label: null, text: "This is general legal information." });
});

test("translated answers keep the same shape", () => {
  // translation.py carries each line's indentation and marker through IndicTrans2 untouched.
  const rows = layout("- **प्रमुख प्रावधान:**\n  - **CrPC § 41**: अधिकारी संतुष्ट है कि गिरफ्तारी आवश्यक है:\n    (i) आगे के अपराध को रोकें;\n    (ii) उचित जांच सुनिश्चित करना;");
  assert.deepEqual(rows.map((r) => [r.depth, r.label]), [[0, "•"], [1, "•"], [2, "(i)"], [2, "(ii)"]]);
  assert.equal(rows[2].text, "आगे के अपराध को रोकें;");
});

test("recognises the enumeration styles statutes use", () => {
  const labels = ["1. a", "12) b", "(1) c", "(a) d", "(iv) e", "a) f", "ii) g", "• h", "* i", "+ j"].map(
    (l) => parseLine(l).marker?.label,
  );
  assert.deepEqual(labels, ["1.", "12)", "(1)", "(a)", "(iv)", "a)", "ii)", "•", "•", "•"]);
});

test("recognises enumerations IndicTrans2 writes in Indian scripts", () => {
  // From a live Hindi answer: the model's (a)–(e) come back as (क)–(ङ).
  const rows = layout("  - अधिकारी संतुष्ट है कि गिरफ्तारी निम्नलिखित के लिए आवश्यक हैः\n    (क) आगे के अपराध को रोकें;\n    (ङ) न्यायालय में व्यक्ति की उपस्थिति सुनिश्चित करना।");
  assert.deepEqual(rows.map((r) => [r.depth, r.label]), [[0, "•"], [1, "(क)"], [1, "(ङ)"]]);
  assert.equal(rows[1].text, "आगे के अपराध को रोकें;");
  const labels = ["(१) a", "२. b", "(ক) c", "क) d", "(க) e"].map((l) => parseLine(l).marker?.label);
  assert.deepEqual(labels, ["(१)", "२.", "(ক)", "क)", "(க)"]);
});

test("leaves ordinary text, emphasis and rules alone", () => {
  for (const l of [
    "**Bold** opening",
    "*Note:* italic opening",
    "---",
    "***",
    "(punishable up to 7 years) applies",
    "(i.e. the officer)",
    "1990. A year, not a list",
    "41(1)(b) of the CrPC",
    "(i",
  ]) {
    assert.equal(parseLine(l).marker, null, l);
  }
});

test("4-space and tab indentation step one level at a time", () => {
  assert.deepEqual(layout("- a\n    - b\n        - c\n    - d\n- e").map((r) => r.depth), [0, 1, 2, 1, 0]);
  assert.deepEqual(layout("1. a\n\t- b").map((r) => r.depth), [0, 1]);
});

test("a paragraph between lists restarts nesting", () => {
  assert.deepEqual(layout("- a\n  - b\nNext paragraph\n  - c").map((r) => r.depth), [0, 1, 0, 0]);
});

test("details-block lines without markers are untouched", () => {
  const line = parseLine("<summary>Validation report</summary>");
  assert.deepEqual(line, { indent: 0, marker: null, text: "<summary>Validation report</summary>" });
});
