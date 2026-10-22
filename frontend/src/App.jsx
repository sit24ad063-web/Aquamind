
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  FileImage,
  FlaskConical,
  Gauge,
  Info,
  LayoutDashboard,
  Leaf,
  LoaderCircle,
  Microscope,
  RefreshCw,
  ShieldCheck,
  Trash2,
  Upload,
  Waves,
  X,
} from "lucide-react";
import "./App.css";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const HISTORY_KEY = "aquamind-coral-analysis-history-v2";
const MAX_FILE_SIZE = 15 * 1024 * 1024;
const MAX_HISTORY = 50;

const NAV_ITEMS = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Analysis", icon: Microscope },
  { label: "Monitoring", icon: Activity },
  { label: "Risk", icon: ShieldCheck },
  { label: "About", icon: Info },
];

function readHistory() {
  try {
    const stored = JSON.parse(
      localStorage.getItem(HISTORY_KEY) || "[]"
    );
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? `${(number * 100).toFixed(1)}%`
    : "â€”";
}

function formatSeconds(value) {
  if (value === null || value === undefined || value === "") {
    return "â€”";
  }

  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)} s` : "â€”";
}

function classLabel(label = "") {
  const normalized = String(label).toLowerCase();

  if (normalized.includes("bleached")) {
    return "Bleached Corals";
  }

  if (normalized.includes("healthy")) {
    return "Healthy Corals";
  }

  return String(label || "Unknown class");
}

function safeDataUrl(value) {
  if (!value || typeof value !== "string") return "";

  if (value.startsWith("data:image/")) return value;

  return `data:image/png;base64,${value}`;
}

function getAnalysisTime(result) {
  return (
    result?.analysis_time_seconds ??
    result?.inference_time_seconds ??
    null
  );
}

function getErrorMessage(data) {
  if (typeof data?.detail === "string") return data.detail;

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg || "Invalid request")
      .join(", ");
  }

  return "The analysis request failed. Check the backend logs.";
}

export default function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(readHistory);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  // Monitor the local FastAPI backend.
  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        const response = await fetch(`${API}/health`);
        if (!response.ok) throw new Error("Health check failed");

        const data = await response.json();

        if (!cancelled) {
          setApiStatus(
            data.model_loaded === false
              ? "model-unavailable"
              : "online"
          );
        }
      } catch {
        if (!cancelled) setApiStatus("offline");
      }
    }

    checkHealth();

    const interval = window.setInterval(checkHealth, 15000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  // Dashboard result button can navigate to the Analysis page.
  useEffect(() => {
    const openAnalysis = () => setActivePage("Analysis");

    window.addEventListener(
      "aquamind-navigate-analysis",
      openAnalysis
    );

    return () => {
      window.removeEventListener(
        "aquamind-navigate-analysis",
        openAnalysis
      );
    };
  }, []);

  // Release temporary image URLs when they are replaced.
  useEffect(() => {
    return () => {
      if (preview.startsWith("blob:")) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const probabilityEntries = useMemo(
    () => Object.entries(result?.probabilities || {}),
    [result]
  );

  const probabilityValues = probabilityEntries
    .map(([, value]) => Number(value))
    .filter(Number.isFinite);

  const probabilityMargin =
    probabilityValues.length >= 2
      ? Math.abs(probabilityValues[0] - probabilityValues[1])
      : null;

  const lowSeparation =
    probabilityMargin !== null && probabilityMargin <= 0.1;

  const isBleached =
    classLabel(result?.predicted_class) === "Bleached Corals";

  const analysisTime = getAnalysisTime(result);

  function chooseFile(nextFile) {
    if (!nextFile) return;

    if (!nextFile.type?.startsWith("image/")) {
      setError("Choose a valid image file, such as JPG, PNG, or WEBP.");
      return;
    }

    if (nextFile.size > MAX_FILE_SIZE) {
      setError("Choose an image smaller than 15 MB.");
      return;
    }

    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
    setError("");
  }

  async function analyzeImage() {
    if (!file) {
      setError("Choose a coral image before starting the analysis.");
      return;
    }

    if (apiStatus === "offline") {
      setError(
        "The API is disconnected. Start your FastAPI backend at http://127.0.0.1:8000."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API}/explain`, {
        method: "POST",
        body,
      });

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The API returned an unreadable response. Check the backend."
        );
      }

      if (!response.ok) {
        throw new Error(getErrorMessage(data));
      }

      if (!data || !data.predicted_class) {
        throw new Error(
          "The API response is missing the predicted_class field."
        );
      }

      setResult(data);

      const entry = {
        id: `${Date.now()}-${file.name}`,
        filename: data.filename || file.name,
        predicted_class: data.predicted_class,
        confidence: data.confidence,
        analysis_time_seconds: getAnalysisTime(data),
        inference_time_seconds: getAnalysisTime(data),
        probabilities: data.probabilities || {},
        visual_features: data.visual_features || null,
        explanation: data.explanation
          ? {
              target_class: data.explanation.target_class,
              target_class_index:
                data.explanation.target_class_index,
              method: data.explanation.method,
              patch_grid_size: data.explanation.patch_grid_size,
              interpretation_note:
                data.explanation.interpretation_note,
            }
          : null,
        created_at: new Date().toISOString(),
        // Blob previews are temporary and are not valid after reload.
        preview: "",
      };

      setHistory((previous) => {
        const next = [entry, ...previous].slice(0, MAX_HISTORY);

        try {
          localStorage.setItem(
            HISTORY_KEY,
            JSON.stringify(next)
          );
        } catch {
          // Keep the live result even if browser storage is unavailable.
        }

        return next;
      });
    } catch (err) {
      const message = err?.message || "Unexpected analysis error.";

      setError(
        message.includes("Failed to fetch")
          ? "Cannot reach the API. Verify that the backend is running and CORS allows your frontend origin."
          : message
      );
    } finally {
      setLoading(false);
    }
  }

  function selectHistory(item) {
    setResult({
      filename: item.filename,
      predicted_class: item.predicted_class,
      confidence: item.confidence,
      probabilities: item.probabilities || {},
      analysis_time_seconds: getAnalysisTime(item),
      inference_time_seconds: getAnalysisTime(item),
      visual_features: item.visual_features || null,
      explanation: item.explanation || null,
      model: "prithivMLmods/Coral-Health",
      device: "Local API",
      warning:
        "Historical prediction only. It does not establish coral presence or ecological health.",
    });

    setFile(null);
    setPreview(item.preview || "");
    setActivePage("Analysis");
    setError("");
  }

  function clearHistory() {
    setHistory([]);

    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch {
      // Ignore browser storage errors.
    }
  }

  function resetAnalysis() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  }

  const pageTitle =
    activePage === "Dashboard"
      ? "Coral intelligence"
      : activePage;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#dashboard"
          onClick={(event) => {
            event.preventDefault();
            setActivePage("Dashboard");
          }}
        >
          <span className="brand-mark">
            <Waves size={23} />
          </span>

          <span className="brand-copy">
            <strong>AquaMind</strong>
            <small>REEF INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="side-nav" aria-label="Main navigation">
          {NAV_ITEMS.map(({ label, icon: Icon }) => (
            <button
              className={`nav-item ${
                activePage === label ? "active" : ""
              }`}
              key={label}
              onClick={() => setActivePage(label)}
              type="button"
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>

              {activePage === label && (
                <ChevronRight
                  className="nav-chevron"
                  size={15}
                />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="api-card">
            <span className={`status-dot ${apiStatus}`} />

            <div>
              <strong>
                {apiStatus === "online"
                  ? "Model API online"
                  : apiStatus === "checking"
                    ? "Checking API"
                    : apiStatus === "model-unavailable"
                      ? "Model unavailable"
                      : "API disconnected"}
              </strong>

              <small>127.0.0.1:8000</small>
            </div>

            <button
              className="icon-button small"
              type="button"
              title="Check API again"
              onClick={async () => {
                setApiStatus("checking");

                try {
                  const response = await fetch(`${API}/health`);

                  if (!response.ok) {
                    throw new Error("Health check failed");
                  }

                  const data = await response.json();

                  setApiStatus(
                    data.model_loaded === false
                      ? "model-unavailable"
                      : "online"
                  );
                } catch {
                  setApiStatus("offline");
                }
              }}
            >
              <RefreshCw size={14} />
            </button>
          </div>

          <p className="sidebar-footnote">
            Research support for clearer reef observations.
          </p>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              AQUAMIND <span>/</span> {activePage.toUpperCase()}
            </div>
            <h1>{pageTitle}</h1>
          </div>

          <div className="topbar-right">
            <span className="local-badge">
              <span className={`status-dot ${apiStatus}`} />
              Local inference
            </span>
            <div className="avatar">AM</div>
          </div>
        </header>

        {activePage === "Dashboard" && (
          <DashboardPage
            file={file}
            preview={preview}
            result={result}
            history={history}
            loading={loading}
            error={error}
            dragActive={dragActive}
            setDragActive={setDragActive}
            chooseFile={chooseFile}
            analyzeImage={analyzeImage}
            resetAnalysis={resetAnalysis}
            selectHistory={selectHistory}
            clearHistory={clearHistory}
            probabilityEntries={probabilityEntries}
            lowSeparation={lowSeparation}
            isBleached={isBleached}
            analysisTime={analysisTime}
          />
        )}

        {activePage === "Analysis" && (
          <AnalysisPage
            result={result}
            preview={preview}
            probabilityEntries={probabilityEntries}
            lowSeparation={lowSeparation}
            isBleached={isBleached}
            analysisTime={analysisTime}
            onBack={() => setActivePage("Dashboard")}
          />
        )}

        {activePage === "Monitoring" && (
          <MonitoringPage history={history} />
        )}

        {activePage === "Risk" && (
          <RiskPage history={history} />
        )}

        {activePage === "About" && <AboutPage />}
      </main>
    </div>
  );
}

function DashboardPage({
  file,
  preview,
  result,
  history,
  loading,
  error,
  dragActive,
  setDragActive,
  chooseFile,
  analyzeImage,
  resetAnalysis,
  selectHistory,
  clearHistory,
  probabilityEntries,
  lowSeparation,
  isBleached,
  analysisTime,
}) {
  return (
    <div className="page-content">
      <section className="welcome-strip">
        <div className="welcome-icon">
          <Leaf size={19} />
        </div>

        <div>
          <strong>Understand what the model sees.</strong>
          <p>
            Run a coral health classification and inspect the visual
            evidence behind its prediction.
          </p>
        </div>

        <span className="welcome-tag">AI Â· EXPLAINABILITY</span>
      </section>

      <section className="summary-grid">
        <StatCard
          icon={FlaskConical}
          label="ANALYSES STORED"
          value={history.length}
          note="Saved in this browser"
        />
        <StatCard
          icon={Cpu}
          label="INFERENCE MODE"
          value="Local CPU"
          note="Connected to your local API"
        />
        <StatCard
          icon={Gauge}
          label="MODEL CLASSES"
          value="02"
          note="Healthy or bleached coral"
        />
        <StatCard
          icon={Clock3}
          label="LATEST RUN"
          value={
            history[0]
              ? formatSeconds(getAnalysisTime(history[0]))
              : "â€”"
          }
          note={
            history[0]
              ? "Most recent analysis"
              : "No analyses yet"
          }
        />
      </section>

      <div className="section-heading">
        <div>
          <span className="eyebrow">ANALYSIS WORKSPACE</span>
          <h2>Inspect a coral sample</h2>
        </div>

        <span className="subtle-label">
          <span className="status-dot online" />
          Explainable image analysis
        </span>
      </div>

      <section className="workspace-grid">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <h3>Image input</h3>
              <p>Upload a clear coral photograph to begin.</p>
            </div>
            <span className="step-number">01</span>
          </div>

          <label
            className={`dropzone ${
              dragActive ? "drag-active" : ""
            } ${preview ? "has-preview" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/gif"
              onChange={(event) => {
                chooseFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />

            {preview ? (
              <>
                <img
                  className="upload-preview"
                  src={preview}
                  alt="Selected coral sample"
                />
                <span className="preview-overlay">
                  <FileImage size={16} />
                  Selected image
                </span>
              </>
            ) : (
              <div className="dropzone-empty">
                <span className="upload-icon">
                  <Upload size={21} />
                </span>
                <strong>Drop your image here</strong>
                <span>
                  or <em>browse files</em>
                </span>
                <small>JPG, PNG, WEBP Â· up to 15 MB</small>
              </div>
            )}
          </label>

          <div className="file-row">
            <div className="file-icon">
              <FileImage size={17} />
            </div>

            <div className="file-info">
              <strong>
                {file?.name ||
                  result?.filename ||
                  "No image selected"}
              </strong>
              <small>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                  : "Choose an image to continue"}
              </small>
            </div>

            {(file || preview) && (
              <button
                className="icon-button"
                onClick={resetAnalysis}
                type="button"
                aria-label="Clear image"
              >
                <X size={16} />
              </button>
            )}
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <button
            className="primary-button analyze-button"
            onClick={analyzeImage}
            disabled={!file || loading}
            type="button"
          >
            {loading ? (
              <>
                <LoaderCircle className="spin" size={17} />
                Analyzing imageâ€¦
              </>
            ) : (
              <>
                Run explainable analysis
                <ArrowRight size={17} />
              </>
            )}
          </button>

          <p className="privacy-note">
            <ShieldCheck size={14} />
            The image is sent to your local API at 127.0.0.1:8000.
          </p>
        </div>

        <div className="panel result-panel">
          <div className="panel-heading">
            <div>
              <h3>Classification result</h3>
              <p>Prediction and confidence returned by the model.</p>
            </div>
            <span className="step-number">02</span>
          </div>

          {result ? (
            <>
              <div
                className={`result-status ${
                  isBleached ? "bleached" : "healthy"
                }`}
              >
                <span className="result-status-icon">
                  {isBleached ? (
                    <AlertCircle size={21} />
                  ) : (
                    <CheckCircle2 size={21} />
                  )}
                </span>

                <div>
                  <small>MODEL PREDICTION</small>
                  <h2>{classLabel(result.predicted_class)}</h2>
                </div>

                <span className="result-confidence">
                  {formatPercent(result.confidence)}
                </span>
              </div>

              {lowSeparation && (
                <div className="uncertainty-notice">
                  <Info size={15} />
                  <span>
                    Low separation between classes. Treat this result
                    as inconclusive.
                  </span>
                </div>
              )}

              <div className="probability-list">
                {probabilityEntries.map(([label, value]) => (
                  <ProbabilityRow
                    key={label}
                    label={label}
                    value={value}
                  />
                ))}
              </div>

              <div className="result-meta-grid">
                <div>
                  <span>Analysis time</span>
                  <strong>{formatSeconds(analysisTime)}</strong>
                </div>
                <div>
                  <span>Compute device</span>
                  <strong>{result.device || "Local API"}</strong>
                </div>
              </div>

              <div className="result-footer">
                <Info size={14} />
                <span>
                  Classification is a model estimate, not a field
                  diagnosis.
                </span>
              </div>

              <button
                className="text-button"
                type="button"
                onClick={() =>
                  window.dispatchEvent(
                    new Event("aquamind-navigate-analysis")
                  )
                }
              >
                View visual explanation
                <ArrowRight size={15} />
              </button>
            </>
          ) : (
            <div className="empty-result">
              <div className="empty-orbit">
                <Microscope size={26} />
              </div>
              <strong>Your analysis will appear here</strong>
              <p>
                Upload a coral image, then run the model to see its
                prediction, confidence scores, and explanation images.
              </p>
              <div className="empty-steps">
                <span>
                  <CheckCircle2 size={14} /> Image input
                </span>
                <span>
                  <Activity size={14} /> Model inference
                </span>
                <span>
                  <Microscope size={14} /> Visual explanation
                </span>
              </div>
            </div>
          )}
        </div>
      </section>

      {result?.explanation && (
        <ExplainabilitySection
          result={result}
          fallbackPreview={preview}
        />
      )}

      <section className="panel history-panel">
        <div className="panel-heading history-heading">
          <div>
            <span className="eyebrow">RECENT ACTIVITY</span>
            <h3>Analysis history</h3>
            <p>Recent results saved in this browser.</p>
          </div>

          {history.length > 0 && (
            <button
              className="text-button danger-text"
              type="button"
              onClick={clearHistory}
            >
              <Trash2 size={14} />
              Clear history
            </button>
          )}
        </div>

        {history.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Analysis time</th>
                  <th>Date</th>
                  <th />
                </tr>
              </thead>

              <tbody>
                {history.slice(0, 8).map((item) => (
                  <tr key={item.id}>
                    <td>
                      <div className="sample-cell">
                        {item.preview ? (
                          <img src={item.preview} alt="" />
                        ) : (
                          <span className="sample-placeholder">
                            <FileImage size={15} />
                          </span>
                        )}
                        <span title={item.filename}>
                          {item.filename}
                        </span>
                      </div>
                    </td>

                    <td>
                      <span
                        className={`class-pill ${
                          classLabel(item.predicted_class) ===
                          "Bleached Corals"
                            ? "pill-bleached"
                            : "pill-healthy"
                        }`}
                      >
                        {classLabel(item.predicted_class)}
                      </span>
                    </td>

                    <td className="numeric-cell">
                      {formatPercent(item.confidence)}
                    </td>

                    <td className="numeric-cell">
                      {formatSeconds(getAnalysisTime(item))}
                    </td>

                    <td className="date-cell">
                      {item.created_at
                        ? new Date(
                            item.created_at
                          ).toLocaleDateString()
                        : "â€”"}
                    </td>

                    <td>
                      <button
                        className="icon-button"
                        title="Open saved result"
                        type="button"
                        onClick={() => selectHistory(item)}
                      >
                        <ArrowRight size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-history">
            Your completed analyses will be listed here.
          </div>
        )}
      </section>
    </div>
  );
}

function ProbabilityRow({ label, value }) {
  const numericValue = Number(value);
  const percentage = Number.isFinite(numericValue)
    ? Math.max(0, Math.min(100, numericValue * 100))
    : 0;

  return (
    <div className="probability-row">
      <div className="probability-label">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="progress-track">
        <div
          className={`progress-fill ${
            String(label).toLowerCase().includes("bleached")
              ? "bleached-fill"
              : ""
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function ExplainabilitySection({ result, fallbackPreview }) {
  const explanation = result.explanation || {};
  const features = result.visual_features || null;
  const images = result.images || {};

  const imageItems = [
    {
      title: "Original image",
      subtitle: "Input sample",
      src: images.original_png_base64
        ? safeDataUrl(images.original_png_base64)
        : fallbackPreview,
    },
    {
      title: "Enhanced image",
      subtitle: "Preprocessed appearance",
      src: images.enhanced_png_base64
        ? safeDataUrl(images.enhanced_png_base64)
        : "",
    },
    {
      title: "Attribution heatmap",
      subtitle: "Regions influencing the selected output",
      src: explanation.heatmap_png_base64
        ? safeDataUrl(explanation.heatmap_png_base64)
        : "",
    },
    {
      title: "Attribution overlay",
      subtitle: "Attribution shown over image context",
      src: explanation.overlay_png_base64
        ? safeDataUrl(explanation.overlay_png_base64)
        : explanation.attribution_map_png_base64
          ? safeDataUrl(explanation.attribution_map_png_base64)
          : "",
    },
  ];

  return (
    <section className="analysis-workspace">
      <div className="section-heading">
        <div>
          <span className="eyebrow">MODEL EXPLAINABILITY</span>
          <h2>Visual analysis</h2>
          <p>
            Compare enhancement, model attribution, and image
            measurements.
          </p>
        </div>

        <span className="method-chip">
          {explanation.method || "Model attribution"}
        </span>
      </div>

      <div className="visual-grid">
        {imageItems.map((item) => (
          <article className="panel visual-card" key={item.title}>
            <div className="visual-card-heading">
              <h3>{item.title}</h3>
              <p>{item.subtitle}</p>
            </div>

            <div className="visual-image-frame">
              {item.src ? (
                <img
                  className="analysis-image"
                  src={item.src}
                  alt={item.title}
                  loading="lazy"
                />
              ) : (
                <div className="image-unavailable">
                  <FileImage size={24} />
                  <span>
                    Image unavailable for this result.
                  </span>
                </div>
              )}
            </div>
          </article>
        ))}
      </div>

      {features && (
        <div className="panel visual-features-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">IMAGE MEASUREMENTS</span>
              <h3>Color, brightness &amp; structure</h3>
              <p>
                Descriptive image statistics, not independent
                diagnoses.
              </p>
            </div>
          </div>

          <div className="feature-metric-grid">
            <FeatureMetric
              label="Image dimensions"
              value={`${features.image_width ?? "â€”"} Ã— ${
                features.image_height ?? "â€”"
              }`}
            />
            <FeatureMetric
              label="Mean brightness"
              value={formatMetric(features.mean_brightness, 4)}
            />
            <FeatureMetric
              label="Brightness variation"
              value={formatMetric(features.brightness_std, 4)}
            />
            <FeatureMetric
              label="Mean edge strength"
              value={formatMetric(features.edge_strength_mean, 5)}
            />
          </div>

          <h4 className="histogram-heading">
            RGB intensity distribution
          </h4>

          <div className="histogram-legend">
            <span className="hist-red">â— Red</span>
            <span className="hist-green">â— Green</span>
            <span className="hist-blue">â— Blue</span>
          </div>

          <div className="histogram-chart">
            {["red", "green", "blue"].map((channel) => {
              const bins =
                features.rgb_histogram_8_bins?.[channel] || [];

              return (
                <div className="histogram-channel" key={channel}>
                  <span
                    className={`histogram-channel-label hist-${channel}`}
                  >
                    {channel.toUpperCase()}
                  </span>

                  <div className="histogram-bars">
                    {bins.map((value, index) => {
                      const numericValue = Number(value);
                      const height = Number.isFinite(numericValue)
                        ? Math.max(
                            2,
                            Math.min(100, numericValue * 100)
                          )
                        : 2;

                      return (
                        <div
                          className="histogram-bin"
                          key={index}
                        >
                          <div
                            className={`histogram-bar hist-${channel}`}
                            style={{ height: `${height}%` }}
                            title={`Bin ${index + 1}: ${numericValue.toFixed(
                              4
                            )}`}
                          />
                          <span>{index}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <p className="feature-disclaimer">
            {features.interpretation_note ||
              "RGB histograms describe pixel intensities. Edge strength summarizes image structure. Neither measurement independently determines coral health."}
          </p>
        </div>
      )}

      <div className="scientific-notice">
        <Info size={18} />
        <div>
          <strong>How to interpret this explanation</strong>
          <p>
            The heatmap highlights image regions that influence the
            model output. It is an attribution visualization, not a
            segmentation mask, and does not prove bleaching.
            Enhancement changes image appearance; RGB histograms and
            edge strength summarize pixel characteristics. This model
            predicts only its trained classes and can misclassify
            unrelated images.
          </p>
        </div>
      </div>
    </section>
  );
}

function formatMetric(value, decimals = 4) {
  if (value === null || value === undefined || value === "") {
    return "â€”";
  }

  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(decimals) : "â€”";
}

function AnalysisPage({
  result,
  preview,
  probabilityEntries,
  lowSeparation,
  isBleached,
  analysisTime,
  onBack,
}) {
  if (!result) {
    return (
      <EmptyPage
        icon={Microscope}
        title="No analysis to inspect"
        description="Return to the dashboard, upload a coral image, and run an analysis first."
        action="Go to dashboard"
        onClick={onBack}
      />
    );
  }

  return (
    <div className="page-content">
      <div className="section-heading">
        <div>
          <span className="eyebrow">DETAILED INSPECTION</span>
          <h2>{result.filename || "Selected sample"}</h2>
          <p>
            Classification output and available visual explanation.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={onBack}
          type="button"
        >
          Back to dashboard
        </button>
      </div>

      <div className="detail-grid">
        <div className="panel detail-image-panel">
          {preview ? (
            <img src={preview} alt="Selected coral sample" />
          ) : (
            <div className="image-unavailable">
              <FileImage size={24} />
              <span>
                Original sample is not available in this saved record.
              </span>
            </div>
          )}
        </div>

        <div className="panel detail-result">
          <span className="eyebrow">PREDICTED CLASS</span>

          <h2
            className={
              isBleached ? "text-bleached" : "text-healthy"
            }
          >
            {classLabel(result.predicted_class)}
          </h2>

          <div className="large-confidence">
            {formatPercent(result.confidence)}
            <small> model score</small>
          </div>

          {lowSeparation && (
            <div className="uncertainty-notice">
              <Info size={15} />
              <span>
                Low separation between classes; interpret as
                inconclusive.
              </span>
            </div>
          )}

          {probabilityEntries.map(([label, value]) => (
            <ProbabilityRow key={label} label={label} value={value} />
          ))}

          <div className="result-meta-grid">
            <div>
              <span>Analysis time</span>
              <strong>{formatSeconds(analysisTime)}</strong>
            </div>
            <div>
              <span>Device</span>
              <strong>{result.device || "Local API"}</strong>
            </div>
          </div>
        </div>
      </div>

      {result.explanation?.heatmap_png_base64 ||
      result.explanation?.overlay_png_base64 ||
      result.explanation?.attribution_map_png_base64 ||
      result.images?.enhanced_png_base64 ? (
        <ExplainabilitySection
          result={result}
          fallbackPreview={preview}
        />
      ) : (
        <div className="scientific-notice">
          <Info size={18} />
          <div>
            <strong>Visual explanation unavailable</strong>
            <p>
              This record does not contain the explanation image data.
              Run a fresh analysis to generate and inspect the
              explanation images.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function MonitoringPage({ history }) {
  const healthy = history.filter(
    (item) =>
      classLabel(item.predicted_class) === "Healthy Corals"
  ).length;

  const bleached = history.filter(
    (item) =>
      classLabel(item.predicted_class) === "Bleached Corals"
  ).length;

  return (
    <div className="page-content">
      <div className="section-heading">
        <div>
          <span className="eyebrow">OBSERVATION LOG</span>
          <h2>Monitoring overview</h2>
          <p>
            A browser-local summary of completed image analyses.
          </p>
        </div>
      </div>

      <section className="summary-grid">
        <StatCard
          icon={Activity}
          label="TOTAL ANALYSES"
          value={history.length}
          note="Saved browser records"
        />
        <StatCard
          icon={CheckCircle2}
          label="HEALTHY CLASS OUTPUTS"
          value={healthy}
          note="Model classifications only"
        />
        <StatCard
          icon={AlertCircle}
          label="BLEACHED CLASS OUTPUTS"
          value={bleached}
          note="Model classifications only"
        />
        <StatCard
          icon={Clock3}
          label="MOST RECENT"
          value={
            history[0]?.created_at
              ? new Date(
                  history[0].created_at
                ).toLocaleDateString()
              : "â€”"
          }
          note="Date of latest saved result"
        />
      </section>

      <div className="scientific-notice">
        <Info size={18} />
        <div>
          <strong>Monitoring limitation</strong>
          <p>
            This page summarizes uploaded images only. It is not a
            live sensor network, geographic reef map, or time-series
            ecological monitoring system. Comparisons across dates
            require consistent sampling and field validation.
          </p>
        </div>
      </div>

      {history.length === 0 && (
        <EmptyPage
          icon={Activity}
          title="No monitoring records yet"
          description="Run your first image analysis to populate this overview."
        />
      )}
    </div>
  );
}

function RiskPage({ history }) {
  const lowConfidence = history.filter(
    (item) => Number(item.confidence) < 0.6
  ).length;

  return (
    <div className="page-content">
      <div className="section-heading">
        <div>
          <span className="eyebrow">
            INTERPRETATION &amp; LIMITATIONS
          </span>
          <h2>Risk and confidence</h2>
          <p>
            Use model outputs cautiously and prioritize field
            verification.
          </p>
        </div>
      </div>

      <div className="risk-banner">
        <div className="risk-banner-icon">
          <ShieldCheck size={23} />
        </div>
        <div>
          <h3>AI output is decision support, not a field diagnosis</h3>
          <p>
            Image quality, lighting, water clarity, camera settings,
            and out-of-distribution images can affect the prediction.
          </p>
        </div>
      </div>

      <section className="summary-grid">
        <StatCard
          icon={Gauge}
          label="LOW-SCORE RECORDS"
          value={lowConfidence}
          note="Saved confidence below 60%"
        />
        <StatCard
          icon={FlaskConical}
          label="TOTAL SAVED RECORDS"
          value={history.length}
          note="Browser-local analysis history"
        />
      </section>

      <div className="panel guidance-panel">
        <h3>Recommended interpretation workflow</h3>
        <ol>
          <li>
            <strong>Check the input.</strong>
            <span>
              Confirm that the image contains coral and is not
              dominated by glare, sediment, or unrelated objects.
            </span>
          </li>
          <li>
            <strong>Review class separation.</strong>
            <span>
              If class probabilities are close, treat the output as
              inconclusive rather than forcing a conclusion.
            </span>
          </li>
          <li>
            <strong>Inspect the attribution map.</strong>
            <span>
              Check which regions influence the model. Attribution is
              not biological segmentation.
            </span>
          </li>
          <li>
            <strong>Validate independently.</strong>
            <span>
              Use expert review, repeat observations, and appropriate
              field measurements before taking action.
            </span>
          </li>
        </ol>
      </div>
    </div>
  );
}

function AboutPage() {
  return (
    <div className="page-content">
      <div className="section-heading">
        <div>
          <span className="eyebrow">PROJECT INFORMATION</span>
          <h2>About AquaMind</h2>
          <p>
            A local-first coral image classification and explainability
            interface.
          </p>
        </div>
      </div>

      <div className="about-hero">
        <div className="about-symbol">
          <Waves size={32} />
        </div>
        <div>
          <span className="eyebrow">
            CORAL HEALTH Â· COMPUTER VISION
          </span>
          <h2>Make model outputs easier to inspect.</h2>
          <p>
            AquaMind connects a browser interface to a local
            image-analysis API and presents predictions alongside
            visual attribution and descriptive pixel measurements.
          </p>
        </div>
      </div>

      <div className="about-grid">
        <InfoCard
          icon={Cpu}
          title="Classification model"
          text="prithivMLmods/Coral-Health, accessed through the local FastAPI service."
        />
        <InfoCard
          icon={Microscope}
          title="Explainability"
          text="Heatmap and overlay outputs help inspect regions influencing the model output."
        />
        <InfoCard
          icon={Activity}
          title="Image measurements"
          text="Brightness, RGB intensity distributions, and edge strength summarize image characteristics."
        />
      </div>

      <div className="scientific-notice">
        <Info size={18} />
        <div>
          <strong>Important scope statement</strong>
          <p>
            The application does not independently verify coral
            presence, identify coral species, measure ocean
            conditions, or establish ecological status. Treat
            predictions as preliminary model outputs that require
            validation.
          </p>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, note }) {
  return (
    <article className="panel stat-card">
      <div className="stat-top">
        <span className="stat-icon">
          <Icon size={18} />
        </span>
        <span className="eyebrow">{label}</span>
      </div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-note">{note}</span>
    </article>
  );
}

function FeatureMetric({ label, value }) {
  return (
    <div className="feature-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InfoCard({ icon: Icon, title, text }) {
  return (
    <article className="panel info-card">
      <span className="info-card-icon">
        <Icon size={19} />
      </span>
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

function EmptyPage({
  icon: Icon,
  title,
  description,
  action,
  onClick,
}) {
  return (
    <div className="panel empty-page">
      <span className="empty-page-icon">
        <Icon size={25} />
      </span>
      <h2>{title}</h2>
      <p>{description}</p>

      {action && (
        <button
          className="primary-button"
          type="button"
          onClick={onClick}
        >
          {action}
          <ArrowRight size={16} />
        </button>
      )}
    </div>
  );
}

