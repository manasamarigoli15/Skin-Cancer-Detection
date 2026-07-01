import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "E:/Cancer Detection Project";
const OUT = path.join(ROOT, "outputs", "Skin_Cancer_Detection_XAI_Presentation.pptx");
const PREVIEW_DIR = path.join(ROOT, "work", "presentations", "skin-cancer-xai-ppt", "tmp", "preview");
const LAYOUT_DIR = path.join(ROOT, "work", "presentations", "skin-cancer-xai-ppt", "tmp", "layout");

const W = 1280;
const H = 720;
const C = {
  ink: "#111111",
  muted: "#555555",
  light: "#EDEDED",
  rule: "#B8BCC4",
  accent: "#FF6B35",
  dark: "#20242A",
  white: "#FFFFFF",
};

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? C.ink,
    fontFace: "Helvetica Neue",
    alignment: style.alignment ?? "left",
  };
  return box;
}

function addTitle(slide, title, kicker = "Skin Cancer Detection + XAI") {
  addText(slide, kicker.toUpperCase(), { left: 54, top: 36, width: 430, height: 30 }, {
    fontSize: 16,
    bold: true,
    color: C.accent,
  });
  addText(slide, title, { left: 54, top: 82, width: 920, height: 72 }, {
    fontSize: 42,
    bold: true,
    color: C.ink,
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 54, top: 164, width: 1172, height: 1.5 },
    fill: C.rule,
    line: { style: "solid", fill: C.rule, width: 0 },
  });
}

function addFooter(slide, num) {
  addText(slide, `HAM10000 Deep Learning Project | ${String(num).padStart(2, "0")}`,
    { left: 54, top: 666, width: 500, height: 24 },
    { fontSize: 14, color: C.muted });
}

function addPanel(slide, position, fill = C.light) {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
}

function addBulletList(slide, items, position, opts = {}) {
  const text = items.map((item) => `- ${item}`).join("\n");
  return addText(slide, text, position, {
    fontSize: opts.fontSize ?? 22,
    color: opts.color ?? C.ink,
  });
}

function addMetric(slide, label, value, x, y, w = 250) {
  addText(slide, value, { left: x, top: y, width: w, height: 64 }, {
    fontSize: 50,
    bold: true,
    color: C.ink,
  });
  addText(slide, label, { left: x, top: y + 68, width: w, height: 34 }, {
    fontSize: 18,
    bold: true,
    color: C.muted,
  });
}

async function addImage(slide, imagePath, position, contentType = "image/png", alt = "Project image") {
  slide.images.add({
    blob: await readImageBlob(imagePath),
    contentType,
    alt,
    fit: "cover",
    position,
  });
}

async function buildDeck() {
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Title
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    addText(s, "Skin Cancer Detection", { left: 70, top: 92, width: 890, height: 84 }, {
      fontSize: 64,
      bold: true,
    });
    addText(s, "Using Deep Learning and Explainable AI", { left: 72, top: 178, width: 940, height: 58 }, {
      fontSize: 34,
      color: C.muted,
    });
    addText(s, "HAM10000 | PyTorch | EfficientNet | ViT | Grad-CAM | Streamlit",
      { left: 74, top: 290, width: 660, height: 44 }, { fontSize: 20, color: C.ink });
    addPanel(s, { left: 72, top: 380, width: 540, height: 118 }, C.dark);
    addText(s, "End-to-end application", { left: 100, top: 410, width: 480, height: 34 }, {
      fontSize: 28,
      bold: true,
      color: C.white,
    });
    addText(s, "From dataset loading to a working web app with prediction heatmaps.",
      { left: 100, top: 452, width: 460, height: 54 }, { fontSize: 17, color: "#E8E8E8" });
    await addImage(s, path.join(ROOT, "data", "raw", "HAM10000_images_part_1", "ISIC_0024306.jpg"),
      { left: 780, top: 92, width: 380, height: 380 }, "image/jpeg", "Dermoscopic lesion sample");
    addFooter(s, 1);
  }

  // 2. Objective
  {
    const s = p.slides.add();
    addTitle(s, "Project objective");
    addText(s, "Build a complete pipeline that can classify dermoscopic lesion images and explain the visual evidence behind each prediction.",
      { left: 64, top: 208, width: 1020, height: 82 }, { fontSize: 30, bold: true });
    addBulletList(s, [
      "Use HAM10000 as the supervised image dataset.",
      "Compare a CNN baseline against a Vision Transformer.",
      "Evaluate with medical-ML-friendly metrics, not accuracy alone.",
      "Expose predictions through a Streamlit app with Grad-CAM evidence."
    ], { left: 90, top: 340, width: 900, height: 190 }, { fontSize: 24 });
    addPanel(s, { left: 1010, top: 232, width: 170, height: 230 }, C.light);
    addText(s, "7", { left: 1050, top: 258, width: 100, height: 70 }, { fontSize: 62, bold: true, alignment: "center" });
    addText(s, "diagnostic classes", { left: 1030, top: 340, width: 130, height: 70 }, { fontSize: 20, bold: true, alignment: "center", color: C.muted });
    addFooter(s, 2);
  }

  // 3. Dataset
  {
    const s = p.slides.add();
    addTitle(s, "Dataset: HAM10000");
    addText(s, "The metadata file defines the exact seven output classes. The model cannot predict diseases outside these labels unless new labeled data is added.",
      { left: 64, top: 190, width: 520, height: 96 }, { fontSize: 22, color: C.muted });
    const labels = [
      ["akiec", "Actinic keratoses"], ["bcc", "Basal cell carcinoma"], ["bkl", "Benign keratosis-like"],
      ["df", "Dermatofibroma"], ["mel", "Melanoma"], ["nv", "Melanocytic nevi"], ["vasc", "Vascular lesions"]
    ];
    labels.forEach((row, i) => {
      const y = 318 + i * 38;
      addText(s, row[0], { left: 72, top: y, width: 80, height: 26 }, { fontSize: 20, bold: true, color: C.accent });
      addText(s, row[1], { left: 160, top: y, width: 360, height: 26 }, { fontSize: 20, color: C.ink });
    });
    s.charts.add("bar", {
      position: { left: 650, top: 218, width: 520, height: 350 },
      categories: ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
      series: [{ name: "Images", values: [327, 514, 1099, 115, 1113, 6705, 142], fill: C.accent }],
      hasLegend: false,
      dataLabels: { showValue: true, position: "outEnd" },
      yAxis: { majorGridlines: { style: "solid", fill: C.rule, width: 1 } },
    });
    addFooter(s, 3);
  }

  // 4. Pipeline
  {
    const s = p.slides.add();
    addTitle(s, "End-to-end workflow");
    const steps = ["Dataset", "Preprocess", "Train", "Evaluate", "Explain", "Deploy"];
    steps.forEach((step, i) => {
      const x = 70 + i * 190;
      addPanel(s, { left: x, top: 270, width: 145, height: 110 }, i === 2 ? "#FFE7DB" : C.light);
      addText(s, String(i + 1), { left: x + 16, top: 286, width: 40, height: 36 }, { fontSize: 28, bold: true, color: C.accent });
      addText(s, step, { left: x + 16, top: 330, width: 110, height: 32 }, { fontSize: 22, bold: true });
      if (i < steps.length - 1) {
        addText(s, ">", { left: x + 154, top: 306, width: 28, height: 50 }, { fontSize: 34, bold: true, color: C.muted, alignment: "center" });
      }
    });
    addText(s, "Every stage is modular: data, models, training, evaluation, utilities, and app code are separated into reusable files.",
      { left: 124, top: 468, width: 1000, height: 72 }, { fontSize: 26, bold: true, alignment: "center" });
    addFooter(s, 4);
  }

  // 5. Preprocessing
  {
    const s = p.slides.add();
    addTitle(s, "Preprocessing and augmentation");
    await addImage(s, path.join(ROOT, "data", "raw", "HAM10000_images_part_1", "ISIC_0024307.jpg"),
      { left: 72, top: 214, width: 350, height: 350 }, "image/jpeg", "Dermoscopic lesion sample");
    addText(s, "Training images are transformed to improve robustness.", { left: 486, top: 212, width: 620, height: 44 }, { fontSize: 28, bold: true });
    addBulletList(s, [
      "Resize every image to 224 x 224 pixels.",
      "Convert image pixels into PyTorch tensors.",
      "Normalize with ImageNet mean and standard deviation.",
      "Apply random flips, rotations, and color jitter only during training.",
      "Keep validation, test, and app inference deterministic."
    ], { left: 514, top: 294, width: 650, height: 230 }, { fontSize: 23 });
    addFooter(s, 5);
  }

  // 6. Models
  {
    const s = p.slides.add();
    addTitle(s, "Model comparison design");
    addPanel(s, { left: 72, top: 230, width: 500, height: 250 }, C.light);
    addText(s, "EfficientNet-B0", { left: 104, top: 266, width: 420, height: 44 }, { fontSize: 34, bold: true });
    addBulletList(s, [
      "CNN baseline",
      "Good accuracy-to-compute balance",
      "Pretrained on ImageNet",
      "Best practical choice for CPU demo"
    ], { left: 108, top: 332, width: 390, height: 120 }, { fontSize: 21 });
    addPanel(s, { left: 708, top: 230, width: 500, height: 250 }, "#F7F7F7");
    addText(s, "Vision Transformer", { left: 740, top: 266, width: 420, height: 44 }, { fontSize: 34, bold: true });
    addBulletList(s, [
      "Patch-token attention model",
      "Learns global image relationships",
      "Useful comparison against CNNs",
      "Usually slower on CPU"
    ], { left: 744, top: 332, width: 410, height: 120 }, { fontSize: 21 });
    addFooter(s, 6);
  }

  // 7. Training
  {
    const s = p.slides.add();
    addTitle(s, "Training setup");
    addMetric(s, "Image size", "224", 92, 220);
    addMetric(s, "Batch size", "32", 346, 220);
    addMetric(s, "Epochs configured", "15", 600, 220);
    addMetric(s, "Classes", "7", 894, 220);
    addText(s, "Training used weighted sampling and weighted cross-entropy to handle the class imbalance visible in HAM10000.",
      { left: 86, top: 438, width: 970, height: 60 }, { fontSize: 27, bold: true });
    addBulletList(s, ["AdamW optimizer", "Cosine learning-rate schedule", "Early stopping", "Best checkpoint saved by validation F1"],
      { left: 108, top: 526, width: 880, height: 86 }, { fontSize: 21 });
    addFooter(s, 7);
  }

  // 8. Training Progress
  {
    const s = p.slides.add();
    addTitle(s, "EfficientNet training progress");
    s.charts.add("line", {
      position: { left: 80, top: 218, width: 760, height: 360 },
      categories: ["1", "2", "3", "4", "5", "6", "7"],
      series: [
        { name: "Val F1", values: [0.3079, 0.3618, 0.4118, 0.3945, 0.4417, 0.4287, 0.4588], fill: C.accent },
        { name: "Val accuracy", values: [0.3160, 0.3846, 0.4810, 0.4431, 0.4604, 0.4405, 0.4717], fill: C.ink },
      ],
      hasLegend: true,
      yAxis: { majorGridlines: { style: "solid", fill: C.rule, width: 1 } },
    });
    addPanel(s, { left: 900, top: 246, width: 260, height: 220 }, C.light);
    addText(s, "Best checkpoint", { left: 930, top: 278, width: 210, height: 32 }, { fontSize: 24, bold: true });
    addText(s, "Epoch 7", { left: 930, top: 326, width: 210, height: 48 }, { fontSize: 42, bold: true, color: C.accent });
    addText(s, "Saved as checkpoints/best_efficientnet.pt", { left: 930, top: 392, width: 200, height: 52 }, { fontSize: 18, color: C.muted });
    addFooter(s, 8);
  }

  // 9. Evaluation metrics
  {
    const s = p.slides.add();
    addTitle(s, "Evaluation results: EfficientNet checkpoint");
    s.charts.add("bar", {
      position: { left: 82, top: 224, width: 720, height: 350 },
      categories: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
      series: [{ name: "Score", values: [46.7, 44.8, 75.7, 46.7, 94.4], fill: C.accent }],
      hasLegend: false,
      dataLabels: { showValue: true, position: "outEnd" },
      yAxis: { min: 0, max: 100, majorGridlines: { style: "solid", fill: C.rule, width: 1 } },
    });
    addText(s, "Result interpretation", { left: 860, top: 236, width: 300, height: 34 }, { fontSize: 26, bold: true });
    addBulletList(s, [
      "ROC-AUC is strong, showing useful class separation.",
      "Accuracy and F1 are limited because training ran on CPU and stopped early.",
      "Recall is high, but precision needs improvement with longer training and tuning."
    ], { left: 862, top: 292, width: 330, height: 210 }, { fontSize: 20 });
    addFooter(s, 9);
  }

  // 10. Confusion Matrix
  {
    const s = p.slides.add();
    addTitle(s, "Confusion matrix");
    await addImage(s, path.join(ROOT, "reports", "figures", "efficientnet_confusion_matrix.png"),
      { left: 92, top: 196, width: 560, height: 430 }, "image/png", "EfficientNet confusion matrix");
    addText(s, "Why this matters", { left: 724, top: 222, width: 360, height: 42 }, { fontSize: 30, bold: true });
    addBulletList(s, [
      "Shows which classes are confused with each other.",
      "More informative than accuracy for imbalanced medical data.",
      "Helps identify classes that need more samples or stronger augmentation.",
      "Supports the comparison between EfficientNet and ViT once both are trained."
    ], { left: 728, top: 298, width: 410, height: 220 }, { fontSize: 22 });
    addFooter(s, 10);
  }

  // 11. Explainability + App
  {
    const s = p.slides.add();
    addTitle(s, "Explainability and web app");
    await addImage(s, path.join(ROOT, "data", "raw", "HAM10000_images_part_1", "ISIC_0024308.jpg"),
      { left: 86, top: 214, width: 330, height: 330 }, "image/jpeg", "Uploaded lesion example");
    addPanel(s, { left: 486, top: 214, width: 330, height: 330 }, "#F6F6F6");
    addText(s, "Grad-CAM", { left: 532, top: 306, width: 240, height: 36 }, { fontSize: 30, bold: true, alignment: "center", color: C.accent });
    addText(s, "heatmap", { left: 532, top: 350, width: 240, height: 34 }, { fontSize: 27, bold: true, alignment: "center", color: C.accent });
    addText(s, "Generated at inference time", { left: 532, top: 408, width: 240, height: 30 }, { fontSize: 18, alignment: "center", color: C.muted });
    addText(s, "Streamlit app output", { left: 880, top: 224, width: 300, height: 38 }, { fontSize: 28, bold: true });
    addBulletList(s, [
      "Upload lesion image",
      "View predicted class",
      "View confidence score",
      "Inspect Grad-CAM overlay",
      "Read short explanation"
    ], { left: 894, top: 292, width: 300, height: 190 }, { fontSize: 22 });
    addFooter(s, 11);
  }

  // 12. Limitations + Next steps
  {
    const s = p.slides.add();
    addTitle(s, "Limitations and next steps");
    addPanel(s, { left: 78, top: 228, width: 500, height: 292 }, C.light);
    addText(s, "Current limitations", { left: 108, top: 262, width: 380, height: 36 }, { fontSize: 28, bold: true });
    addBulletList(s, [
      "Not a medical diagnostic device",
      "HAM10000 is class-imbalanced",
      "CPU training limits final accuracy",
      "Grad-CAM explains influence, not correctness"
    ], { left: 112, top: 322, width: 400, height: 148 }, { fontSize: 21 });
    addPanel(s, { left: 704, top: 228, width: 500, height: 292 }, "#F7F7F7");
    addText(s, "Recommended improvements", { left: 734, top: 262, width: 410, height: 36 }, { fontSize: 28, bold: true });
    addBulletList(s, [
      "Train longer with GPU support",
      "Tune augmentation and class weights",
      "Complete ViT comparison",
      "Add external validation data"
    ], { left: 738, top: 322, width: 400, height: 148 }, { fontSize: 21 });
    addText(s, "The application already demonstrates a complete research-to-app workflow.",
      { left: 120, top: 578, width: 1040, height: 42 }, { fontSize: 28, bold: true, alignment: "center" });
    addFooter(s, 12);
  }

  // 13. Closing
  {
    const s = p.slides.add();
    s.background.fill = C.dark;
    addText(s, "Thank you", { left: 80, top: 170, width: 760, height: 90 }, { fontSize: 70, bold: true, color: C.white });
    addText(s, "Skin Cancer Detection Using Deep Learning and Explainable AI", { left: 84, top: 282, width: 890, height: 44 }, { fontSize: 28, color: "#E8E8E8" });
    addText(s, "Demo: streamlit run app/streamlit_app.py", { left: 86, top: 410, width: 720, height: 40 }, { fontSize: 24, color: C.accent, bold: true });
    addText(s, "Checkpoint: checkpoints/best_efficientnet.pt", { left: 86, top: 462, width: 720, height: 34 }, { fontSize: 22, color: "#E8E8E8" });
  }

  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });

  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(PREVIEW_DIR, `${stem}.png`), await p.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(path.join(PREVIEW_DIR, "deck-montage.webp"), await p.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  console.log(OUT);
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});



