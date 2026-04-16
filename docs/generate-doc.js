const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ExternalHyperlink, Bookmark,
  TabStopType, TabStopPosition,
} = require("docx");

// ── Shared styling constants ──────────────────────────────────

const BRAND_BLUE = "1A5276";
const BRAND_DARK = "2C3E50";
const ACCENT_GREEN = "27AE60";
const ACCENT_ORANGE = "E67E22";
const ACCENT_RED = "C0392B";
const LIGHT_GRAY = "F4F6F7";
const MID_GRAY = "D5D8DC";
const TABLE_HEADER_BG = "1A5276";
const TABLE_ALT_BG = "EBF5FB";

const border = { style: BorderStyle.SINGLE, size: 1, color: MID_GRAY };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

const PAGE_WIDTH = 12240;
const MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN; // 9360

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: TABLE_HEADER_BG, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 20 })] })],
  });
}

function cell(text, width, opts = {}) {
  const { bold, color, shading, align } = opts;
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: shading ? { fill: shading, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!bold, color: color || "2C3E50", font: "Arial", size: 20 })],
    })],
  });
}

function makeTable(headers, rows, colWidths) {
  const tableRows = [
    new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
    ...rows.map((row, ri) =>
      new TableRow({
        children: row.map((c, ci) => {
          const isObj = typeof c === "object" && c !== null && !Array.isArray(c);
          return cell(
            isObj ? c.text : String(c),
            colWidths[ci],
            {
              shading: ri % 2 === 1 ? TABLE_ALT_BG : undefined,
              bold: isObj ? c.bold : false,
              color: isObj ? c.color : undefined,
            }
          );
        }),
      })
    ),
  ];
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows,
  });
}

function heading(level, text) {
  const sizes = { 1: 36, 2: 30, 3: 26 };
  const spacing = { 1: { before: 360, after: 200 }, 2: { before: 280, after: 160 }, 3: { before: 200, after: 120 } };
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: spacing[level],
    children: [new TextRun({ text, bold: true, font: "Arial", size: sizes[level], color: BRAND_DARK })],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, before: opts.before || 0 },
    alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Arial", size: opts.size || 22, color: opts.color || "2C3E50", bold: !!opts.bold, italics: !!opts.italics })],
  });
}

function multiPara(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, before: opts.before || 0 },
    alignment: opts.align || AlignmentType.LEFT,
    children: runs.map(r => new TextRun({ text: r.text, font: "Arial", size: r.size || 22, color: r.color || "2C3E50", bold: !!r.bold, italics: !!r.italics })),
  });
}

function spacer(pts = 200) {
  return new Paragraph({ spacing: { before: pts } });
}

// Bullet list items using numbering config
function bullet(text, ref = "bullets", level = 0) {
  return new Paragraph({
    numbering: { reference: ref, level },
    spacing: { after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 22, color: "2C3E50" })],
  });
}

function boldBullet(label, desc, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [
      new TextRun({ text: label + " ", font: "Arial", size: 22, color: "2C3E50", bold: true }),
      new TextRun({ text: desc, font: "Arial", size: 22, color: "2C3E50" }),
    ],
  });
}

function codeBlock(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: "F7F9F9", type: ShadingType.CLEAR },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Consolas", size: 18, color: "566573" })],
  });
}

function divider() {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND_BLUE, space: 1 } },
  });
}

// ── BUILD DOCUMENT ──────────────────────────────────────────

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BRAND_DARK },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: BRAND_DARK },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: BRAND_DARK },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
    characterStyles: [
      { id: "Hyperlink", name: "Hyperlink", run: { color: "2980B9", underline: { type: "single" } } },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }, {
          level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } },
        }],
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [

    // ════════════════════════════════════════════════════
    //  COVER PAGE
    // ════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: 15840 },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      children: [
        spacer(3000),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "AuthentiGuard", font: "Arial", size: 72, bold: true, color: BRAND_BLUE })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "AI Content Authenticity Detection Platform", font: "Arial", size: 32, color: BRAND_DARK })],
        }),
        divider(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "Technical Architecture & Operations Manual", font: "Arial", size: 28, color: "566573" })],
        }),
        spacer(600),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Version 1.0  |  April 2026", font: "Arial", size: 22, color: "7F8C8D" })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "CONFIDENTIAL", font: "Arial", size: 22, bold: true, color: ACCENT_RED })],
        }),
        spacer(2000),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Model Version: 3.2-g2-removed-product-output", font: "Consolas", size: 20, color: "7F8C8D" })],
        }),
      ],
    },

    // ════════════════════════════════════════════════════
    //  TABLE OF CONTENTS + MAIN CONTENT
    // ════════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: 15840 },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND_BLUE, space: 1 } },
            children: [
              new TextRun({ text: "AuthentiGuard  ", font: "Arial", size: 18, bold: true, color: BRAND_BLUE }),
              new TextRun({ text: "Technical Architecture & Operations Manual", font: "Arial", size: 18, color: "7F8C8D" }),
            ],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 18, color: "7F8C8D" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "7F8C8D" }),
              new TextRun({ text: "  |  Confidential", font: "Arial", size: 18, color: "7F8C8D" }),
            ],
          })],
        }),
      },
      children: [

        // ── 1. EXECUTIVE SUMMARY ──
        heading(1, "1. Executive Summary"),

        heading(2, "1.1 What Is AuthentiGuard?"),
        para("AuthentiGuard is an AI content authenticity detection platform that determines whether digital content (text, images, audio, video, and code) was produced by a human or generated by artificial intelligence. It provides a confidence-scored verdict with explainable evidence, enabling organizations to verify the provenance of content at scale."),

        heading(2, "1.2 Why It Exists"),
        para("The proliferation of generative AI (ChatGPT, Midjourney, DALL-E, Suno, etc.) has created a trust crisis: educators cannot verify student work, publishers cannot trust submitted articles, and enterprises cannot audit their own internal communications. AuthentiGuard fills this gap with a multi-layered detection ensemble that goes beyond single-model approaches to deliver production-grade reliability."),

        heading(2, "1.3 Key Capabilities"),
        boldBullet("Multi-content detection:", "Text, image, audio, video, and code analysis pipelines"),
        boldBullet("Ensemble AI detection:", "3-layer text ensemble (perplexity + stylometry + DeBERTa-v3-small) with meta-classifier"),
        boldBullet("Reliability-gated decisions:", "Tri-zone labeling (AI / UNCERTAIN / HUMAN) with 0.70/0.30 thresholds"),
        boldBullet("Explainable verdicts:", "Per-sentence scoring with evidence signals (hedge words, contractions, vocabulary formality)"),
        boldBullet("Enterprise-grade infrastructure:", "JWT auth, per-tier rate limiting, audit logging, GDPR compliance"),
        boldBullet("Production observability:", "Prometheus metrics, Grafana dashboards, automated drift detection, webhook alerting"),
        boldBullet("Resilient architecture:", "Redis AOF persistence, graceful ML fallback, bounded backpressure on all async paths"),

        heading(2, "1.4 Headline Accuracy (Measured, Held-Out Test Sets)"),
        makeTable(
          ["Metric", "v1 Test (n=2,000)", "v2 Test (n=3,482)"],
          [
            ["F1 Score", { text: "0.9945", bold: true, color: ACCENT_GREEN }, { text: "0.9529", bold: true, color: ACCENT_GREEN }],
            ["Precision", "0.9960", "0.9243"],
            ["Recall", "0.9930", "0.9832"],
            ["AUROC", "0.9977", "0.9767"],
            ["UNCERTAIN Rate", "3.25% (65/2000)", "7.1% (247/3482)"],
          ],
          [3120, 3120, 3120]
        ),
        para("v2 test includes adversarial subsets (paraphrased AI, humanized AI, AI-ified human). All numbers are post-fit on weights [0.20, 0.35, 0.45] with AI threshold 0.41.", { italics: true, size: 20 }),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 2. SYSTEM ARCHITECTURE ──
        heading(1, "2. System Architecture"),

        heading(2, "2.1 High-Level Architecture"),
        para("AuthentiGuard follows a monorepo structure with clear separation between frontend, backend API, AI detection pipelines, and infrastructure:"),

        makeTable(
          ["Component", "Technology", "Purpose"],
          [
            ["Frontend", "Next.js 14 + React", "User-facing dashboard, analysis submission, result visualization"],
            ["API Backend", "FastAPI (Python 3.11+)", "REST API, auth, job orchestration, report generation"],
            ["Task Queue", "Celery + Redis", "Async job processing with priority queues"],
            ["AI Detectors", "PyTorch + HuggingFace", "ML inference pipelines (text, image, audio, video, code)"],
            ["Database", "PostgreSQL 16", "Users, jobs, results, audit logs, webhooks"],
            ["Cache / Broker", "Redis 7", "Rate limiting, session cache, Celery message broker"],
            ["Object Storage", "MinIO (S3-compatible)", "File uploads, generated reports"],
            ["Monitoring", "Prometheus + Grafana", "Metrics collection, dashboards, alerting"],
            ["Reverse Proxy", "Caddy 2 (prod)", "Auto-HTTPS, HTTP/3, routing"],
            ["ML Tracking", "MLflow", "Experiment tracking, model registry"],
          ],
          [2200, 2560, 4600]
        ),

        heading(2, "2.2 Repository Structure"),
        codeBlock("authentiguard/"),
        codeBlock("  frontend/          Next.js 14 app (React, TypeScript)"),
        codeBlock("  backend/           FastAPI app + Celery workers"),
        codeBlock("    app/"),
        codeBlock("      api/v1/        REST endpoints (routes.py, deps.py)"),
        codeBlock("      core/          Config, database, redis, security, metrics"),
        codeBlock("      middleware/     Rate limiting, audit logging"),
        codeBlock("      workers/       Celery tasks (text, image, audio, video, webhook, drift, alerting)"),
        codeBlock("      models/        SQLAlchemy ORM models"),
        codeBlock("      schemas/       Pydantic request/response schemas"),
        codeBlock("      services/      S3, upload, report generation services"),
        codeBlock("      observability/ Prediction logging (JSONL)"),
        codeBlock("  ai/                AI detection pipelines"),
        codeBlock("    text_detector/   Text ensemble (layers, evaluation, checkpoints)"),
        codeBlock("    image_detector/  EfficientNet-B4 + ViT-B/16 pipeline"),
        codeBlock("    audio_detector/  Audio analysis pipeline"),
        codeBlock("    video_detector/  Video frame analysis pipeline"),
        codeBlock("    code_detector/   AST + CodeBERT pipeline"),
        codeBlock("    ensemble_engine/ Cross-content dispatcher + routing"),
        codeBlock("  infra/             Docker, K8s, Terraform, monitoring configs"),
        codeBlock("  scripts/           Training, evaluation, drift computation"),
        codeBlock("  datasets/          Training/test data (DVC-tracked)"),

        heading(2, "2.3 Request Flow"),
        para("A typical text analysis request flows through the system as follows:"),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Client submits text via POST /api/v1/analyze/text (or file upload)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "FastAPI validates input, creates DetectionJob in PostgreSQL (status=PENDING)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Job dispatched to Celery text queue with tier-based priority (free=1, pro=5, enterprise=9)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Worker picks up job, loads text from DB or S3, runs TextDetector.analyze()", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "3-layer ensemble runs: L1 perplexity, L2 stylometry, L3 DeBERTa semantic analysis", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Scores combined via Stage 2 LR meta-classifier (or Stage 1 weighted average fallback)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Reliability-gated decision: score >= 0.70 = AI, score <= 0.30 = HUMAN, else UNCERTAIN", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "DetectionResult written to PostgreSQL, prediction logged to JSONL, webhook fired if configured", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Client polls GET /api/v1/jobs/{id} until status=COMPLETED, then reads full result", font: "Arial", size: 22, color: "2C3E50" })],
        }),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 3. AI DETECTION PIPELINE (TEXT) ──
        heading(1, "3. AI Detection Pipeline (Text)"),
        para("The text detection pipeline is the most mature and thoroughly evaluated detector. It uses a multi-layer ensemble architecture where each layer provides an independent signal about whether text was AI-generated."),

        heading(2, "3.1 Layer Architecture"),
        makeTable(
          ["Layer", "Name", "Technique", "Status"],
          [
            ["L1", "Perplexity", "GPT-2 language model perplexity + burstiness + low-perplexity fraction", "Active"],
            ["L2", "Stylometry", "spaCy NLP: sentence variance, hedge words, TTR, contractions, em-dash rate", "Active"],
            ["L3", "Semantic", "DeBERTa-v3-small (44M params) fine-tuned on adversarial corpus", "Active"],
            ["L4", "Adversarial", "Adversarial robustness layer (DeBERTa variant)", { text: "Not trained", color: ACCENT_ORANGE }],
          ],
          [800, 1600, 4360, 2600]
        ),

        heading(3, "3.1.1 Layer 1: Perplexity Analysis"),
        para("Uses a pretrained GPT-2 model (no fine-tuning) to measure how predictable text is. AI-generated text has lower perplexity (more predictable) than human writing. The layer computes three signals:"),
        boldBullet("Perplexity signal (0.45 weight):", "Sigmoid-mapped distance from AI mean (36) vs human mean (85)"),
        boldBullet("Burstiness signal (0.25 weight):", "Variance of per-sentence perplexity (humans are more variable)"),
        boldBullet("Low-perplexity fraction (0.30 weight):", "Proportion of sentences with very low perplexity"),

        heading(3, "3.1.2 Layer 2: Stylometric Fingerprinting"),
        para("Uses spaCy (en_core_web_sm) plus hand-tuned heuristics to detect AI writing style. Seven independent signals are averaged:"),
        bullet("Sentence length variance (low variance = AI)"),
        bullet("AI hedge word frequency (furthermore, moreover, delve, leverage, etc.)"),
        bullet("Human casual word absence (actually, basically, gonna, lol, etc.)"),
        bullet("Type-Token Ratio / vocabulary diversity"),
        bullet("Sentence-initial word diversity"),
        bullet("Em-dash overuse rate"),
        bullet("Comma usage rate"),

        heading(3, "3.1.3 Layer 3: Semantic Analysis (DeBERTa-v3-small)"),
        para("The most powerful layer. A DeBERTa-v3-small model (44M parameters, microsoft/deberta-v3-small) fine-tuned on an adversarial-augmented corpus. Uses sliding-window inference for long texts. Checkpoint: ai/text_detector/checkpoints/transformer_v3_hard/phase1 (267 MB)."),
        para("Note: Historical documentation incorrectly referred to this as \"DistilBERT.\" The actual architecture is DeBERTa-v3-small.", { italics: true }),

        heading(2, "3.2 Score Combination"),
        para("Scores from active layers are combined in a two-stage process:"),

        heading(3, "Stage 1: Fixed-Weight Combiner (Fallback)"),
        para("Weighted average with grid-searched weights on validation data (9,471 evaluations):"),
        makeTable(
          ["Layer", "Weight", "Contribution"],
          [
            ["L1 (Perplexity)", "0.20", "Baseline signal from language model predictability"],
            ["L2 (Stylometry)", "0.35", "Independent stylistic features (hedge words, contractions)"],
            ["L3 (Semantic)", "0.45", "Deep semantic analysis from fine-tuned transformer"],
          ],
          [2500, 1500, 5360]
        ),
        para("AI threshold: 0.41 (grid-searched). Val F1 at this configuration: 0.9969."),

        heading(3, "Stage 2: Learned Meta-Classifier (Primary)"),
        para("A LogisticRegression stacking model + isotonic CalibratedClassifierCV trained on 2,000 validation rows. Loaded automatically if both checkpoint files exist:"),
        bullet("meta_classifier.joblib (578 bytes) - LogisticRegression"),
        bullet("meta_calibrator.joblib (1,043 bytes) - Isotonic calibration bundle"),
        para("LR coefficients reveal true layer importance: L3=8.769 (dominant), L1=2.752 (useful residual), L2=0.080 (near-zero). Calibrated probabilities concentrate near 0 and 1, collapsing the UNCERTAIN zone from 247 to 1 sample on v2 test."),
        para("If either meta artifact is missing, the system silently falls back to Stage 1 weights with no crash and no config change required."),

        heading(2, "3.3 Decision Policy"),
        para("A reliability-gated 3-zone decision policy converts the continuous score to a label:"),
        makeTable(
          ["Zone", "Score Range", "Label", "Meaning"],
          [
            [{ text: "AI Zone", bold: true, color: ACCENT_RED }, ">= 0.70", { text: "AI", bold: true, color: ACCENT_RED }, "High confidence the text is AI-generated"],
            [{ text: "Uncertain Zone", bold: true, color: ACCENT_ORANGE }, "0.31 - 0.69", { text: "UNCERTAIN", bold: true, color: ACCENT_ORANGE }, "Insufficient confidence to make a definitive call"],
            [{ text: "Human Zone", bold: true, color: ACCENT_GREEN }, "<= 0.30", { text: "HUMAN", bold: true, color: ACCENT_GREEN }, "High confidence the text is human-written"],
          ],
          [2000, 1800, 2000, 3560]
        ),
        para("Gating rule G1: Texts under 50 words are forced to UNCERTAIN (L1/L2 have insufficient signal on short inputs)."),
        para("Note: Gate G2 (L2-L3 disagreement) was removed because it fired on ~48% of inputs, killing AI recall to 2.3%.", { italics: true }),

        heading(2, "3.4 DevFallback Detector"),
        para("When ML models fail to load (missing checkpoints, GPU issues, CI environments), the system falls back to _DevFallbackDetector: a pure-heuristic detector that mirrors L2 stylometry with 10 weighted signals. Different thresholds apply (AI > 0.55, HUMAN < 0.40). This ensures the system never fully crashes, though accuracy is significantly reduced in fallback mode."),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 4. OTHER DETECTION PIPELINES ──
        heading(1, "4. Other Detection Pipelines"),
        para("While the text detector is the most mature, AuthentiGuard includes detection pipelines for all major content types:"),

        makeTable(
          ["Content Type", "Architecture", "Status"],
          [
            ["Image", "EfficientNet-B4 + ViT-B/16 ensemble, GAN fingerprint analysis, FFT spectral, texture analysis", "Implemented"],
            ["Audio", "Spectrogram analysis + audio fingerprinting", "Implemented"],
            ["Video", "Frame-by-frame analysis with temporal consistency", "Implemented"],
            ["Code", "AST structural analysis + CodeBERT semantic model", "Implemented"],
          ],
          [1500, 5860, 2000]
        ),
        para("Each detector follows the same pattern: a Celery worker with lazy-loaded singleton detector, BaseDetectionWorker integration, and unified DetectorOutput format via the ensemble engine dispatcher."),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 5. API REFERENCE ──
        heading(1, "5. API Reference"),

        heading(2, "5.1 Authentication"),
        makeTable(
          ["Endpoint", "Method", "Description"],
          [
            ["/api/v1/auth/register", "POST", "Create new user account (email, password, full name)"],
            ["/api/v1/auth/login", "POST", "Authenticate and receive JWT access + refresh tokens"],
            ["/api/v1/auth/refresh", "POST", "Rotate refresh token and get new access token"],
            ["/api/v1/auth/logout", "POST", "Revoke refresh token"],
            ["/api/v1/auth/forgot-password", "POST", "Initiate password reset flow"],
            ["/api/v1/auth/reset-password", "POST", "Complete password reset with token"],
          ],
          [3500, 1200, 4660]
        ),
        para("Access tokens expire in 15 minutes (HS256 JWT). Refresh tokens last 30 days, stored in Redis, rotated on every use (stolen tokens have a single-use window)."),

        heading(2, "5.2 Analysis Endpoints"),
        makeTable(
          ["Endpoint", "Method", "Description"],
          [
            ["/api/v1/analyze/text", "POST", "Submit text for AI detection (paste or inline text)"],
            ["/api/v1/analyze/upload", "POST", "Upload file (.txt, .pdf, .docx, images, audio, video, code)"],
            ["/api/v1/analyze/url", "POST", "Fetch content from URL (SSRF-protected) and route to detector"],
            ["/api/v1/jobs/{id}", "GET", "Poll job status (PENDING / PROCESSING / COMPLETED / FAILED)"],
            ["/api/v1/jobs/{id}/result", "GET", "Get full detection result with scores, evidence, layer breakdown"],
            ["/api/v1/jobs/{id}/report", "GET", "Download PDF or JSON forensic report"],
          ],
          [3500, 1200, 4660]
        ),

        heading(2, "5.3 Dashboard & Webhooks"),
        makeTable(
          ["Endpoint", "Method", "Description"],
          [
            ["/api/v1/dashboard/stats", "GET", "Usage statistics (total jobs, avg score, content type breakdown)"],
            ["/api/v1/webhooks", "POST", "Create webhook subscription (notify on job completion)"],
            ["/api/v1/webhooks", "GET", "List user webhooks"],
            ["/api/v1/webhooks/{id}", "PATCH", "Update webhook URL or events"],
            ["/api/v1/webhooks/{id}", "DELETE", "Delete webhook subscription"],
          ],
          [3500, 1200, 4660]
        ),

        heading(2, "5.4 System Endpoints"),
        makeTable(
          ["Endpoint", "Method", "Description"],
          [
            ["/health", "GET", "Health check (DB + Redis + detector mode)"],
            ["/metrics", "GET", "Prometheus metrics endpoint (prometheus_client format)"],
          ],
          [3500, 1200, 4660]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 6. SECURITY ARCHITECTURE ──
        heading(1, "6. Security Architecture"),

        heading(2, "6.1 Authentication & Authorization"),
        boldBullet("JWT Tokens:", "HS256 algorithm, 15-minute access token TTL, 32+ character secret key enforced at startup"),
        boldBullet("Refresh Rotation:", "Every refresh token use invalidates the old token and issues a new one (stored in Redis). Stolen tokens have a single-use window."),
        boldBullet("Password Hashing:", "bcrypt with 72-byte truncation (passlib). No plaintext storage."),
        boldBullet("Tier-Based Access:", "Three user tiers (free, pro, enterprise) with different rate limits and priority levels"),

        heading(2, "6.2 Rate Limiting"),
        para("Sliding-window rate limiter using Redis sorted sets. Applied per-user (authenticated) or per-IP (anonymous):"),
        makeTable(
          ["Tier", "Limit", "Window"],
          [
            ["Free / Anonymous", "10 requests", "1 minute"],
            ["Pro", "100 requests", "1 minute"],
            ["Enterprise", "1,000 requests", "1 minute"],
          ],
          [3120, 3120, 3120]
        ),
        para("When Redis is unavailable, rate limiting fails CLOSED (503) to prevent abuse of a public API without protection."),

        heading(2, "6.3 Security Headers"),
        bullet("X-Frame-Options: DENY (clickjacking protection)"),
        bullet("X-Content-Type-Options: nosniff"),
        bullet("X-XSS-Protection: 1; mode=block"),
        bullet("Referrer-Policy: strict-origin-when-cross-origin"),
        bullet("HSTS: max-age=63072000; includeSubDomains; preload (production only)"),
        bullet("Content-Security-Policy: restrictive default-src self (production only)"),

        heading(2, "6.4 Input Validation & Data Protection"),
        boldBullet("Request size limit:", "10 MB hard limit (middleware-enforced)"),
        boldBullet("File upload limits:", "50 MB configurable, extension whitelist per content type"),
        boldBullet("SSRF protection:", "URL analysis fetches are SSRF-protected"),
        boldBullet("Parameter redaction:", "Audit log middleware redacts sensitive query params (password, token, secret, key)"),
        boldBullet("Encryption at rest:", "Fernet key for PII encryption (ENCRYPTION_KEY setting)"),
        boldBullet("CORS:", "Explicit origin whitelist (not wildcard)"),

        heading(2, "6.5 Secrets Management"),
        para("Production deployments use Docker secrets (file-backed, never in env vars or image layers):"),
        bullet("MinIO root credentials: /run/secrets/minio_root_user, minio_root_password"),
        bullet("S3 app credentials: /run/secrets/s3_app_access_key, s3_app_secret_key"),
        bullet("Config.py reads *_FILE env vars and populates credentials at boot"),
        para("Development uses plain .env file (committed to .gitignore, never to source control)."),

        heading(2, "6.6 Audit Logging"),
        para("Every API request is logged to the audit_logs table (PostgreSQL) via AuditLogMiddleware:"),
        bullet("Records: user ID, IP, action, resource type, status code, latency, error message"),
        bullet("Bounded backpressure: max 200 concurrent audit writes per process"),
        bullet("10-second timeout per write to prevent hung DB from blocking slots"),
        bullet("Drops increment AUDIT_LOG_DROPPED Prometheus counter (never blocks the request path)"),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 7. INFRASTRUCTURE & DEPLOYMENT ──
        heading(1, "7. Infrastructure & Deployment"),

        heading(2, "7.1 Docker Compose Services"),
        para("The platform runs as a Docker Compose stack with the following services:"),
        makeTable(
          ["Service", "Image", "Dev Port", "Prod Port"],
          [
            ["postgres", "postgres:16-alpine", "5432", "Internal only"],
            ["redis", "redis:7-alpine", "6379", "Internal only"],
            ["minio", "minio/minio:latest", "9000, 9001", "Internal only"],
            ["backend", "Custom (FastAPI)", "8000", "Via Caddy"],
            ["worker", "Custom (Celery)", "N/A", "N/A"],
            ["beat", "Custom (Celery Beat)", "N/A", "N/A"],
            ["flower", "mher/flower:2.0", "5555", "Internal only"],
            ["frontend", "Custom (Next.js 14)", "3000", "Via Caddy"],
            ["prometheus", "prom/prometheus:v2.53.0", "9090", "Internal only"],
            ["grafana", "grafana/grafana:11.1.0", "3001", "Internal only"],
            ["mlflow", "ghcr.io/mlflow/mlflow:v2.13.0", "5000", "Internal only"],
            ["caddy", "caddy:2-alpine (prod only)", "N/A", "80, 443"],
          ],
          [2000, 2860, 1800, 2700]
        ),

        heading(2, "7.2 Celery Task Queue"),
        para("Five dedicated queues with priority support (0-9, higher = process first):"),
        makeTable(
          ["Queue", "Content Type", "Workers"],
          [
            ["text", "Text + Code detection", "Default workers"],
            ["image", "Image detection", "Dedicated workers"],
            ["audio", "Audio detection", "Dedicated workers"],
            ["video", "Video detection (heaviest)", "Dedicated workers (separate)"],
            ["webhook", "Webhook delivery", "Lightweight workers"],
          ],
          [2000, 4360, 3000]
        ),
        para("Key configuration:"),
        bullet("task_acks_late=True: Jobs acknowledged only after completion (no lost tasks)"),
        bullet("worker_prefetch_multiplier=1: Process one task at a time (GPU workloads)"),
        bullet("task_soft_time_limit=120s, task_time_limit=180s: Bounded execution"),
        bullet("worker_max_tasks_per_child=1000: Memory leak safety net"),
        bullet("task_reject_on_worker_lost=True: Requeue on worker crash"),

        heading(2, "7.3 Periodic Tasks (Celery Beat)"),
        makeTable(
          ["Task", "Schedule", "Purpose"],
          [
            ["cleanup_stuck_jobs", "Every 5 minutes", "Clean up jobs stuck in PROCESSING or PENDING too long"],
            ["check_health", "Every 60 seconds", "Alert on detector fallback, high failure rate, queue depth, Redis down"],
            ["run_daily_drift", "Daily at 02:00 UTC", "Compute daily metrics + PSI drift detection"],
          ],
          [3000, 2360, 4000]
        ),

        heading(2, "7.4 Production Resource Limits"),
        makeTable(
          ["Service", "Memory Limit", "Memory Reserved", "Notes"],
          [
            ["backend", "1 GB", "512 MB", "FastAPI + middleware"],
            ["worker", "7.2 GB", "6 GB", "ML models (DeBERTa + GPT-2 + spaCy)"],
            ["beat", "256 MB", "N/A", "Scheduler only, heuristic mode"],
            ["postgres", "2 GB", "512 MB", "Database"],
            ["redis", "768 MB", "256 MB", "512 MB maxmemory, allkeys-lru eviction"],
            ["prometheus", "512 MB", "128 MB", "30-day retention"],
            ["grafana", "256 MB", "N/A", "Dashboard rendering"],
          ],
          [2000, 2200, 2200, 2960]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 8. OBSERVABILITY & MONITORING ──
        heading(1, "8. Observability & Monitoring"),

        heading(2, "8.1 Prometheus Metrics"),
        para("All metrics are registered in the default prometheus_client registry and scraped at GET /metrics every 15 seconds."),

        heading(3, "HTTP Metrics (main.py)"),
        makeTable(
          ["Metric", "Type", "Labels", "Description"],
          [
            ["http_request_duration_seconds", "Histogram", "method, path, status", "HTTP request latency (p50/p95/p99)"],
            ["http_requests_total", "Counter", "method, path, status", "Total HTTP requests"],
          ],
          [3200, 1200, 2360, 2600]
        ),

        heading(3, "Business Metrics (core/metrics.py)"),
        makeTable(
          ["Metric", "Type", "Labels", "Description"],
          [
            ["detection_duration_seconds", "Histogram", "content_type, detector_mode", "End-to-end detection latency"],
            ["detection_score", "Histogram", "content_type", "AI detection score distribution"],
            ["detection_jobs_total", "Counter", "status", "Total jobs by outcome (completed/failed/timeout)"],
            ["rate_limit_hits_total", "Counter", "tier", "Rate-limited requests by tier"],
            ["model_load_duration_seconds", "Gauge", "N/A", "ML model load time at startup"],
            ["detector_fallback_active", "Gauge", "N/A", "1 = heuristic fallback, 0 = ML active"],
            ["meta_classifier_fallback_active", "Gauge", "N/A", "1 = fixed weights, 0 = LR meta active"],
            ["celery_queue_depth", "Gauge", "queue", "Messages in each Celery queue"],
            ["audit_log_dropped_total", "Counter", "N/A", "Audit writes dropped (backpressure)"],
            ["stuck_jobs_cleaned_total", "Counter", "reason", "Stuck jobs cleaned up"],
          ],
          [3200, 1200, 2360, 2600]
        ),

        heading(2, "8.2 Grafana Dashboard"),
        para("Auto-provisioned dashboard (authentiguard-ops) with 12 panels:"),
        makeTable(
          ["Panel", "Type", "Key Query"],
          [
            ["Request Rate (RPS)", "Timeseries", "rate(http_requests_total[1m])"],
            ["HTTP Latency (p50/p95/p99)", "Timeseries", "histogram_quantile on http_request_duration_seconds"],
            ["Detection Duration", "Timeseries", "histogram_quantile on detection_duration_seconds"],
            ["Celery Queue Depth", "Timeseries (bars)", "celery_queue_depth by queue"],
            ["Error Rate", "Timeseries", "5xx rate / total + job failure rate"],
            ["Detection Jobs", "Timeseries", "detection_jobs_total by status"],
            ["Detector Fallback", "Stat (red/green)", "detector_fallback_active (ML Active vs FALLBACK)"],
            ["Meta Classifier Health", "Stat", "meta_classifier_fallback_active"],
            ["Audit Log Dropped", "Stat", "increase(audit_log_dropped_total[1h])"],
            ["Score Distribution", "Heatmap", "detection_score_bucket"],
            ["Model Load Duration", "Stat", "model_load_duration_seconds"],
            ["Rate Limit Hits", "Timeseries (bars)", "rate_limit_hits_total by tier"],
          ],
          [2800, 1800, 4760]
        ),
        para("Access: localhost:3001 (dev) | Internal-only via Caddy (prod). Default credentials: admin/admin (dev), must be set via GRAFANA_PASSWORD (prod)."),

        heading(2, "8.3 Alerting"),
        para("Two-layer alerting system:"),
        heading(3, "Layer 1: Celery Beat Health Check (every 60s)"),
        bullet("Detector fallback active: CRITICAL alert + webhook"),
        bullet("Job failure rate > 10% (5-min window): WARNING alert + webhook"),
        bullet("Queue depth > 100 messages: WARNING alert + webhook"),
        bullet("Redis unreachable: CRITICAL alert + webhook"),

        heading(3, "Layer 2: Drift Detection (daily at 02:00 UTC)"),
        bullet("PSI >= 0.25: CRITICAL - Significant drift, retraining required"),
        bullet("PSI >= 0.10: WARNING - Moderate drift, monitor closely"),
        para("All alerts emit structlog entries (always available in pod logs) AND post to ALERT_WEBHOOK_URL if configured (Slack-compatible JSON payload). 15-minute per-key cooldown prevents spam."),

        heading(2, "8.4 Prediction Logging"),
        para("Every detection result is logged to JSONL files (logs/predictions/{date}.jsonl) with:"),
        bullet("Timestamp, model version, content type, final score, label, confidence"),
        bullet("Per-layer scores (L1, L2, L3)"),
        bullet("Meta probability (if Stage 2 active)"),
        bullet("1% sample includes full input text (for drift analysis)"),
        para("These logs feed the daily drift detection pipeline."),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 9. DATA PERSISTENCE & RESILIENCE ──
        heading(1, "9. Data Persistence & Resilience"),

        heading(2, "9.1 Redis Persistence"),
        para("Redis 7 is configured with dual persistence for maximum durability:"),
        boldBullet("AOF (Append-Only File):", "appendonly yes, appendfsync everysec - sub-second durability, logs every write"),
        boldBullet("RDB Snapshots:", "save 60 1, save 300 10 (prod) - fast recovery from point-in-time snapshots"),
        boldBullet("Volume:", "redis_data:/data - persists across container restarts"),
        para("Production also sets: maxmemory 512mb, maxmemory-policy allkeys-lru, tcp-keepalive 60, timeout 300."),

        heading(2, "9.2 Redis Client Resilience"),
        para("The async Redis client (backend/app/core/redis.py) is hardened against transient failures:"),
        bullet("ExponentialBackoff retry: 3 retries with cap=2s, base=0.1s on ConnectionError/TimeoutError"),
        bullet("Socket timeouts: 5s connect, 5s read (no hanging connections)"),
        bullet("Health check interval: 30s pings on idle connections (detect stale sockets)"),
        bullet("reset_redis() recovery: tears down singleton and rebuilds pool on persistent failure"),

        heading(2, "9.3 Graceful Degradation"),
        makeTable(
          ["Failure", "Behavior", "Impact"],
          [
            ["ML models fail to load", "DevFallbackDetector activates automatically", "Reduced accuracy; CRITICAL alert fires"],
            ["Meta-classifier missing", "Stage 1 fixed-weight combiner used", "Slightly wider UNCERTAIN zone; no crash"],
            ["Redis unavailable", "Rate limiter returns 503; audit continues", "API rejects requests (fail-closed for security)"],
            ["PostgreSQL slow", "Audit writes dropped after 200 in-flight cap", "Audit gap; request path unaffected"],
            ["Webhook endpoint down", "Best-effort POST with 5s timeout", "Alert only in structlog; no retry flood"],
            ["Drift reference missing", "Drift task logs warning and skips PSI", "No drift data; metrics task still runs"],
          ],
          [2600, 3760, 3000]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 10. DRIFT DETECTION ──
        heading(1, "10. Distribution Drift Detection"),
        para("AuthentiGuard includes automated drift detection to alert when the production score distribution shifts away from the training distribution, which could indicate model degradation or a change in input population."),

        heading(2, "10.1 How It Works"),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Daily at 02:00 UTC, the drift worker reads the day's prediction log (JSONL)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Computes aggregate metrics (mean score, label distribution, percentiles)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Calculates Population Stability Index (PSI) against training_distribution.json", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Writes results to metrics/daily_metrics.json and metrics/drift.json (atomic writes)", font: "Arial", size: 22, color: "2C3E50" })],
        }),
        new Paragraph({
          numbering: { reference: "numbers", level: 0 },
          spacing: { after: 60 },
          children: [new TextRun({ text: "Fires alerts via structlog + webhook when PSI exceeds thresholds", font: "Arial", size: 22, color: "2C3E50" })],
        }),

        heading(2, "10.2 PSI Thresholds"),
        makeTable(
          ["PSI Range", "Classification", "Alert Level", "Action"],
          [
            ["< 0.10", "Stable", "None", "No action needed"],
            ["0.10 - 0.25", "Moderate", { text: "WARNING", bold: true, color: ACCENT_ORANGE }, "Monitor closely; distribution shifting"],
            ["> 0.25", "Significant", { text: "CRITICAL", bold: true, color: ACCENT_RED }, "Investigation or retraining required"],
          ],
          [1800, 2000, 2000, 3560]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 11. CONFIGURATION REFERENCE ──
        heading(1, "11. Configuration Reference"),
        para("All configuration is via environment variables, validated by Pydantic Settings at startup. Missing required vars cause immediate boot failure with a clear error message."),

        heading(2, "11.1 Required Environment Variables"),
        makeTable(
          ["Variable", "Example", "Purpose"],
          [
            ["APP_SECRET_KEY", "(random 64+ chars)", "Application secret for CSRF/misc"],
            ["DATABASE_URL", "postgresql+asyncpg://user:pass@host/db", "PostgreSQL connection string"],
            ["REDIS_URL", "redis://:pass@host:6379/0", "Redis connection string"],
            ["JWT_SECRET_KEY", "(random 32+ chars)", "JWT signing key (HS256)"],
            ["CELERY_BROKER_URL", "redis://:pass@host:6379/0", "Celery message broker"],
            ["CELERY_RESULT_BACKEND", "redis://:pass@host:6379/1", "Celery result backend"],
            ["ENCRYPTION_KEY", "(Fernet key)", "At-rest encryption key"],
            ["POSTGRES_PASSWORD", "(strong password)", "PostgreSQL password"],
            ["REDIS_PASSWORD", "(strong password)", "Redis auth password"],
          ],
          [3200, 2860, 3300]
        ),

        heading(2, "11.2 Key Optional Variables"),
        makeTable(
          ["Variable", "Default", "Purpose"],
          [
            ["APP_ENV", "development", "Environment: development | staging | production"],
            ["DETECTOR_MODE", "ml", "ml = full ML ensemble, heuristic = lightweight fallback"],
            ["ALERT_WEBHOOK_URL", "(empty)", "Slack-compatible webhook for production alerts"],
            ["RATE_LIMIT_FREE_TIER", "10", "Free tier requests/minute"],
            ["RATE_LIMIT_PRO_TIER", "100", "Pro tier requests/minute"],
            ["RATE_LIMIT_ENTERPRISE_TIER", "1000", "Enterprise tier requests/minute"],
            ["ALLOW_DB_CREATE_ALL", "false", "Dev-only: auto-create tables (blocked in production)"],
            ["CORS_ORIGINS", "http://localhost:3000", "Comma-separated allowed origins"],
          ],
          [3600, 2600, 3160]
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 12. KNOWN LIMITATIONS ──
        heading(1, "12. Known Limitations & Risks"),

        heading(2, "12.1 Detection Limitations"),
        boldBullet("Short text (<50 words):", "Forced to UNCERTAIN. Insufficient signal for reliable classification."),
        boldBullet("Adversarial attacks:", "Heavily paraphrased or back-translated AI text can evade detection. The adversarial layer (L4) is not yet trained."),
        boldBullet("Mixed content:", "Text that combines human and AI writing (e.g., human-edited AI draft) may produce misleading scores."),
        boldBullet("Language:", "Optimized for English. Other languages will produce unreliable results."),
        boldBullet("Evolving AI models:", "New LLMs (GPT-5, Claude 4, etc.) may produce text with different characteristics than the training corpus."),

        heading(2, "12.2 Infrastructure Limitations"),
        boldBullet("Worker metrics gap:", "Prometheus only scrapes the backend API process. Worker-emitted metrics (detection duration, score distribution, queue depth) exist in worker memory but are not scraped by Prometheus. Dashboard panels for these metrics will show \"No data\" until a Pushgateway or worker HTTP endpoint is added."),
        boldBullet("Single worker scaling:", "Current deployment uses a single Celery worker. Scaling to N workers means N independent cooldown windows for alerts (possible duplicate notifications)."),
        boldBullet("No HA Redis:", "Single Redis instance. Redis failure degrades rate limiting, caching, and task dispatch simultaneously."),
        boldBullet("No automated failover:", "No Kubernetes auto-healing. Docker Compose restart=unless-stopped provides basic recovery."),

        heading(2, "12.3 Accuracy Caveats"),
        boldBullet("v2 precision trade:", "Post-fit precision dropped 3.5 points (0.96 to 0.92) in exchange for 15-point recall gain. Higher false-positive rate on adversarial subsets."),
        boldBullet("In-sample calibration:", "ECE 0.0000 is measured in-sample (isotonic fit on same data). Real-world calibration will be higher."),
        boldBullet("AUROC gap on v2:", "0.0065 AUROC regression from Stage 1 to Stage 2 on adversarial test set. Accepted as bias-variance trade for calibrated probabilities."),

        new Paragraph({ children: [new PageBreak()] }),

        // ── 13. NEXT STEPS / ROADMAP ──
        heading(1, "13. Next Steps & Roadmap"),

        heading(2, "13.1 Immediate (Pre-Launch)"),
        boldBullet("Fix worker metrics scraping:", "Add Prometheus Pushgateway or start_http_server() in Celery workers so detection_duration, detection_score, queue_depth, and model_load_duration are actually collected."),
        boldBullet("Set ALERT_WEBHOOK_URL:", "Configure Slack/PagerDuty webhook in production .env so alerts reach a human."),
        boldBullet("Run Alembic migrations:", "Production must boot with alembic upgrade head, not create_all."),
        boldBullet("Load test:", "Verify the 7.2 GB worker memory limit holds under concurrent load with real ML models."),

        heading(2, "13.2 Short-Term (Post-Launch, 1-3 Months)"),
        boldBullet("Train L4 adversarial layer:", "Fine-tune DeBERTa checkpoint on adversarial corpus to improve robustness against paraphrased/back-translated AI text."),
        boldBullet("Multi-language support:", "Train or fine-tune detection models for top 5 non-English languages."),
        boldBullet("Redis Sentinel/Cluster:", "Add HA Redis for production resilience."),
        boldBullet("Horizontal worker scaling:", "Deploy multiple Celery workers with autoscaling based on queue depth."),
        boldBullet("C2PA integration:", "Add Content Credentials (C2PA/CAI) verification for images/video provenance."),

        heading(2, "13.3 Medium-Term (3-6 Months)"),
        boldBullet("Browser extension:", "Ship Chrome/Firefox extension for inline detection on any webpage."),
        boldBullet("API marketplace listing:", "Publish API on RapidAPI/AWS Marketplace for self-service enterprise access."),
        boldBullet("Batch processing:", "Bulk file upload with CSV/ZIP support for enterprise document scanning."),
        boldBullet("Model versioning & A/B testing:", "MLflow model registry integration for canary deployments of new detector versions."),
        boldBullet("Kubernetes migration:", "Move from Docker Compose to K8s for auto-healing, rolling deploys, and resource isolation."),

        heading(2, "13.4 Long-Term (6-12 Months)"),
        boldBullet("Real-time streaming detection:", "WebSocket API for live content monitoring (chat, voice, video streams)."),
        boldBullet("Federated learning:", "Enterprise-private model fine-tuning without sharing training data."),
        boldBullet("Regulatory compliance pack:", "EU AI Act compliance toolkit, SOC 2 audit trail, GDPR right-to-erasure for stored content."),
        boldBullet("Multi-modal fusion:", "Cross-modal detection (e.g., AI-generated image with AI-generated caption treated as a unit)."),

        new Paragraph({ children: [new PageBreak()] }),

        // ── APPENDIX A: STARTUP COMMANDS ──
        heading(1, "Appendix A: Startup Commands"),

        heading(2, "Development"),
        codeBlock("# Clone and configure"),
        codeBlock("git clone <repo-url> && cd authentiguard"),
        codeBlock("cp .env.example .env  # Fill in all required variables"),
        codeBlock(""),
        codeBlock("# Start all services"),
        codeBlock("docker compose up -d"),
        codeBlock(""),
        codeBlock("# Run database migrations"),
        codeBlock("docker compose exec backend alembic upgrade head"),
        codeBlock(""),
        codeBlock("# Verify"),
        codeBlock("curl http://localhost:8000/health"),
        codeBlock("# Grafana: http://localhost:3001 (admin/admin)"),
        codeBlock("# Prometheus: http://localhost:9090"),
        codeBlock("# Flower: http://localhost:5555"),
        codeBlock("# MLflow: http://localhost:5000"),

        heading(2, "Production"),
        codeBlock("# Create secrets directory"),
        codeBlock("mkdir -p secrets"),
        codeBlock("echo 'minio-root-user' > secrets/minio_root_user"),
        codeBlock("echo 'minio-root-pass' > secrets/minio_root_password"),
        codeBlock("echo 'app-s3-key' > secrets/s3_app_access_key"),
        codeBlock("echo 'app-s3-secret' > secrets/s3_app_secret_key"),
        codeBlock(""),
        codeBlock("# Configure production env"),
        codeBlock("cp .env.production.example .env.production"),
        codeBlock("# Edit: APP_ENV=production, strong passwords, ALERT_WEBHOOK_URL"),
        codeBlock(""),
        codeBlock("# Deploy"),
        codeBlock("docker compose -f docker-compose.prod.yml up -d"),
        codeBlock("docker compose -f docker-compose.prod.yml exec backend alembic upgrade head"),

        heading(2, "Running Tests"),
        codeBlock("# Backend unit tests (242 tests)"),
        codeBlock("cd backend && python -m pytest tests/ -v"),
        codeBlock(""),
        codeBlock("# AI detector tests (24 tests)"),
        codeBlock("cd ai && python -m pytest text_detector/tests/ -v"),
        codeBlock(""),
        codeBlock("# E2E smoke tests"),
        codeBlock("cd backend && python -m pytest tests/e2e/ -v"),

        spacer(400),
        divider(),
        para("End of Document", { align: AlignmentType.CENTER, italics: true, color: "7F8C8D" }),
        para("AuthentiGuard Technical Architecture & Operations Manual v1.0", { align: AlignmentType.CENTER, size: 20, color: "7F8C8D" }),
      ],
    },
  ],
});

// ── Generate DOCX ──────────────────────────────────────────

Packer.toBuffer(doc).then(buffer => {
  const outPath = "D:\\Authentic\\authentiguard\\docs\\AuthentiGuard-Technical-Manual.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Document generated: " + outPath);
  console.log("Size: " + (buffer.length / 1024).toFixed(1) + " KB");
});
