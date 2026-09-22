import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, Cell, ReferenceLine
} from "recharts";

/* ---------------------------------------------------------------------
   MODEL — a linear SVM over standardized features, trained on the exact
   preprocessing in the uploaded notebook (label + one-hot style encoding,
   dropna, '3+' -> 4). The notebook's own unscaled SVC failed to converge
   in reasonable time on this income data, so features are standardized
   before fitting here — same linear-SVM approach, tractable in-browser.
--------------------------------------------------------------------- */
const FEATURE_NAMES = ["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"];
const SCALER_MEAN = [0.8171296296296297,0.6412037037037037,0.8564814814814815,0.7986111111111112,0.14583333333333334,5541.206018518518,1563.9118518255555,146.47916666666666,342.22222222222223,0.8634259259259259,1.0069444444444444];
const SCALER_SCALE = [0.38656021265900337,0.4796472808849818,1.2295567665828435,0.40103790883280305,0.35293904887702954,5917.2570023697135,2682.850524365114,82.74122719106438,64.33726719885826,0.34339714088047296,0.7890734819245563];
const SVM_COEF = [5.934663729184608e-05,4.545957995991112e-05,-3.137994563318092e-05,-2.013303944047029e-05,-1.8005061058493088e-05,-1.4107975808988482e-05,-8.703765098230787e-05,-1.4292572885141186e-06,2.7620478171575242e-05,0.6868531432374783,7.947273028968695e-06];
const SVM_INTERCEPT = 0.7268827927299308;
const PROB_A = -1.8029434371766195;
const PROB_B = -0.4950900459009858;

const NUMERIC_STATS = {
  ApplicantIncome:{min:150,max:81000,mean:5364.23,median:3859},
  CoapplicantIncome:{min:0,max:33837,mean:1581.09,median:1084.5},
  LoanAmount:{min:9,max:600,mean:144.74,median:128},
  Loan_Amount_Term:{min:36,max:480,mean:342.05,median:360}
};

const METRICS = {
  train_accuracy:0.8056, test_accuracy:0.8333, precision:0.8378, recall:0.9394, f1:0.8857,
  roc_auc:0.8343, confusion_matrix:[[9,6],[2,31]], n_train:432, n_test:48, n_total:480,
  n_approved:332, n_declined:148
};

const ROC_CURVE = [{fpr:0,tpr:0},{fpr:0,tpr:0.03},{fpr:0,tpr:0.121},{fpr:0.067,tpr:0.121},{fpr:0.067,tpr:0.333},{fpr:0.133,tpr:0.333},{fpr:0.133,tpr:0.818},{fpr:0.2,tpr:0.818},{fpr:0.2,tpr:0.848},{fpr:0.267,tpr:0.848},{fpr:0.267,tpr:0.879},{fpr:0.333,tpr:0.879},{fpr:0.333,tpr:0.939},{fpr:0.867,tpr:0.939},{fpr:0.867,tpr:1},{fpr:1,tpr:1}];

const APPROVAL_BY_CREDIT = { "No credit history": 0.1, "Has credit history": 0.793 };
const APPROVAL_BY_AREA = { "Rural": 0.612, "Semiurban": 0.78, "Urban": 0.653 };
const APPROVAL_BY_EDU = { "Not graduate": 0.629, "Graduate": 0.708 };
const APPROVAL_BY_MARRIED = { "Not married": 0.621, "Married": 0.73 };
const OVERALL_RATE = 0.6917;

const DEFAULT_INPUT = {
  Gender: 1, Married: 1, Dependents: 0, Education: 1, Self_Employed: 0,
  ApplicantIncome: 4000, CoapplicantIncome: 1100, LoanAmount: 130,
  Loan_Amount_Term: 360, Credit_History: 1, Property_Area: 1
};

function sigmoid(x) { return 1 / (1 + Math.exp(x)); }

function predict(input) {
  let decision = SVM_INTERCEPT;
  const contributions = [];
  FEATURE_NAMES.forEach((f, i) => {
    const z = (input[f] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    const contrib = z * SVM_COEF[i];
    decision += contrib;
    contributions.push({ feature: f, contrib });
  });
  const probability = sigmoid(PROB_A * decision + PROB_B);
  return { decision, probability, contributions };
}

const FEATURE_LABELS = {
  Gender: "Gender", Married: "Marital status", Dependents: "Dependents", Education: "Education",
  Self_Employed: "Self-employed", ApplicantIncome: "Applicant income", CoapplicantIncome: "Co-applicant income",
  LoanAmount: "Loan amount", Loan_Amount_Term: "Loan term", Credit_History: "Credit history",
  Property_Area: "Property area"
};

/* ---------------------------------------------------------------------
   ANIMATED BACKGROUND — a document-scanner sweep over faint ledger
   rules, colored green when the live application reads approved,
   clay-red when it reads declined.
--------------------------------------------------------------------- */
function LedgerField({ probability }) {
  const canvasRef = useRef(null);
  const probRef = useRef(probability);
  probRef.current = probability;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let raf, t = 0;
    let w = 0, h = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    function draw() {
      const p = probRef.current;
      t += 1;
      ctx.clearRect(0, 0, w, h);

      const green = [94, 143, 107];
      const red = [168, 69, 59];
      const mix = green.map((g, i) => Math.round(red[i] + (g - red[i]) * p));

      // faint ledger rules
      const ruleGap = 34;
      ctx.strokeStyle = "rgba(150,150,130,0.045)";
      ctx.lineWidth = 1;
      for (let y = 20; y < h; y += ruleGap) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      // faint vertical margin line
      ctx.strokeStyle = "rgba(180,150,90,0.05)";
      ctx.beginPath();
      ctx.moveTo(w * 0.09, 0);
      ctx.lineTo(w * 0.09, h);
      ctx.stroke();

      // scanning sweep band
      const cycle = 620;
      const pos = (t * 1.7) % (cycle + h);
      const bandY = pos - h * 0.3;
      const grad = ctx.createLinearGradient(0, bandY - 90, 0, bandY + 90);
      grad.addColorStop(0, `rgba(${mix[0]},${mix[1]},${mix[2]},0)`);
      grad.addColorStop(0.5, `rgba(${mix[0]},${mix[1]},${mix[2]},${0.10 + p * 0.02})`);
      grad.addColorStop(1, `rgba(${mix[0]},${mix[1]},${mix[2]},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, bandY - 90, w, 180);

      ctx.strokeStyle = `rgba(${mix[0]},${mix[1]},${mix[2]},0.22)`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(0, bandY);
      ctx.lineTo(w, bandY);
      ctx.stroke();

      // drifting ledger motes
      for (let i = 0; i < 20; i++) {
        const seed = i * 71.3;
        const px = ((seed * 3.3 + t * 0.25) % (w + 40)) - 20;
        const py = h * 0.08 + ((Math.sin(seed + t * 0.0025) + 1) / 2) * h * 0.84;
        ctx.beginPath();
        ctx.rect(px, py, 3, 1.2);
        ctx.fillStyle = `rgba(${mix[0]},${mix[1]},${mix[2]},0.14)`;
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    }
    draw();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />;
}

/* ---------------------------------------------------------------------
   UI PRIMITIVES
--------------------------------------------------------------------- */
function Toggle({ label, value, onChange, options }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 13, color: "#A9B0A3", marginBottom: 7, fontFamily: "Georgia, serif" }}>{label}</div>
      <div style={{ display: "flex", gap: 6 }}>
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className="loan-toggle"
            style={{
              flex: 1, padding: "8px 10px", fontSize: 12.5, borderRadius: 2, cursor: "pointer",
              fontFamily: "Georgia, serif", transition: "all 150ms ease",
              background: value === opt.value ? "#B8935A" : "transparent",
              color: value === opt.value ? "#14180F" : "#8A9086",
              border: `1px solid ${value === opt.value ? "#B8935A" : "#2A3324"}`
            }}
          >{opt.label}</button>
        ))}
      </div>
    </div>
  );
}

function NumSlider({ name, value, onChange, prefix }) {
  const stats = NUMERIC_STATS[name];
  const pct = ((value - stats.min) / (stats.max - stats.min)) * 100;
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: "#A9B0A3", fontFamily: "Georgia, serif" }}>{FEATURE_LABELS[name]}</span>
        <span style={{ fontSize: 13.5, color: "#E8E4D8", fontVariantNumeric: "tabular-nums", fontFamily: "'Courier New', monospace" }}>
          {prefix}{Math.round(value).toLocaleString()}
        </span>
      </div>
      <div style={{ position: "relative", height: 22, display: "flex", alignItems: "center" }}>
        <div style={{
          position: "absolute", left: 0, right: 0, height: 3, borderRadius: 2,
          background: "linear-gradient(90deg, #3E4A36 0%, #3E4A36 " + pct + "%, #1A2016 " + pct + "%, #1A2016 100%)"
        }} />
        <input
          type="range"
          min={stats.min} max={stats.max} step={Math.max(1, Math.round((stats.max - stats.min) / 300))}
          value={value}
          onChange={(e) => onChange(name, parseFloat(e.target.value))}
          style={{
            WebkitAppearance: "none", appearance: "none", width: "100%", background: "transparent",
            position: "relative", zIndex: 2, cursor: "pointer", height: 22
          }}
          className="loan-slider"
        />
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: "#141C16", border: "1px solid #263229", borderRadius: 3, padding: "14px 16px", flex: "1 1 120px" }}>
      <div style={{ fontSize: 21, color: accent || "#E8E4D8", fontFamily: "'Courier New', monospace", fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: 11.5, color: "#7C8578", marginTop: 3 }}>{label}</div>
      {sub && <div style={{ fontSize: 10.5, color: "#525A4C", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontSize: 12.5, color: "#7C8578", marginBottom: 14, fontFamily: "Georgia, serif", fontStyle: "italic" }}>{children}</div>;
}

/* ---------------------------------------------------------------------
   MAIN APP
--------------------------------------------------------------------- */
export default function LoanApprovalPlatform() {
  const [input, setInput] = useState(DEFAULT_INPUT);

  const handleChange = useCallback((name, value) => {
    setInput((prev) => ({ ...prev, [name]: value }));
  }, []);

  const result = useMemo(() => predict(input), [input]);
  const { probability, contributions } = result;
  const approved = probability >= 0.5;

  const importanceData = useMemo(() => {
    return [...contributions]
      .map((c) => ({ name: FEATURE_LABELS[c.feature], value: c.contrib }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 7);
  }, [contributions]);

  const creditBarData = Object.entries(APPROVAL_BY_CREDIT).map(([name, value]) => ({ name, value }));
  const areaBarData = Object.entries(APPROVAL_BY_AREA).map(([name, value]) => ({ name, value }));
  const eduBarData = Object.entries(APPROVAL_BY_EDU).map(([name, value]) => ({ name, value }));
  const marriedBarData = Object.entries(APPROVAL_BY_MARRIED).map(([name, value]) => ({ name, value }));

  const m = METRICS;
  const cm = m.confusion_matrix;

  return (
    <div style={{
      fontFamily: "'Iowan Old Style', 'Palatino Linotype', Georgia, serif",
      background: "#0B0F0A", color: "#E8E4D8", minHeight: "100vh", position: "relative", overflow: "hidden"
    }}>
      <style>{`
        .loan-slider::-webkit-slider-thumb {
          -webkit-appearance: none; appearance: none;
          width: 15px; height: 15px; border-radius: 50%;
          background: #E8E4D8; border: 2px solid #0B0F0A;
          box-shadow: 0 0 0 1px #B8935A; cursor: pointer; margin-top: -6px;
        }
        .loan-slider::-webkit-slider-runnable-track { height: 3px; background: transparent; }
        .loan-slider::-moz-range-thumb {
          width: 13px; height: 13px; border-radius: 50%;
          background: #E8E4D8; border: 2px solid #B8935A; cursor: pointer;
        }
        .loan-slider::-moz-range-track { height: 3px; background: transparent; }
        .loan-toggle:hover { border-color: #B8935A !important; }
        @media (max-width: 860px) {
          .lp-grid { grid-template-columns: 1fr !important; }
          .lp-hero { grid-template-columns: 1fr !important; }
          .lp-toggles { grid-template-columns: 1fr 1fr !important; }
        }
      `}</style>

      <div style={{ position: "absolute", inset: 0, height: 560, opacity: 0.9 }}>
        <LedgerField probability={probability} />
        <div style={{
          position: "absolute", inset: 0,
          background: "linear-gradient(180deg, rgba(11,15,10,0.30) 0%, rgba(11,15,10,0.55) 55%, #0B0F0A 100%)"
        }} />
      </div>

      <div style={{ position: "relative", zIndex: 1, maxWidth: 1120, margin: "0 auto", padding: "56px 28px 80px" }}>

        {/* Hero */}
        <div style={{ marginBottom: 46 }}>
          <div style={{ fontSize: 13, color: "#B8935A", letterSpacing: "0.02em", marginBottom: 10 }}>
            {m.n_total} applications on file · linear support-vector model
          </div>
          <h1 style={{
            fontSize: "clamp(30px, 4.4vw, 46px)", lineHeight: 1.12, margin: 0, fontWeight: 500,
            maxWidth: 660, color: "#F0ECDE"
          }}>
            Eleven lines on an application, read the way the ledger reads them.
          </h1>
          <p style={{ fontSize: 15.5, color: "#9CA290", maxWidth: 580, marginTop: 16, lineHeight: 1.6 }}>
            Fill in an application below. The sweep overhead follows the read the same way the model does —
            it settles green on approval, clay-red on decline. One field carries almost the entire decision;
            you'll feel it the moment you touch it.
          </p>
        </div>

        {/* Hero: inputs + verdict */}
        <div className="lp-hero" style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: 28, marginBottom: 54 }}>
          <div style={{ background: "rgba(20,28,22,0.72)", border: "1px solid #263229", borderRadius: 4, padding: "26px 28px", backdropFilter: "blur(6px)" }}>
            <SectionLabel>Applicant details</SectionLabel>

            <div className="lp-toggles" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20, marginBottom: 6 }}>
              <Toggle label="Gender" value={input.Gender} onChange={(v) => handleChange("Gender", v)}
                options={[{ value: 0, label: "Female" }, { value: 1, label: "Male" }]} />
              <Toggle label="Marital status" value={input.Married} onChange={(v) => handleChange("Married", v)}
                options={[{ value: 0, label: "Single" }, { value: 1, label: "Married" }]} />
              <Toggle label="Education" value={input.Education} onChange={(v) => handleChange("Education", v)}
                options={[{ value: 0, label: "Not grad." }, { value: 1, label: "Graduate" }]} />
              <Toggle label="Self-employed" value={input.Self_Employed} onChange={(v) => handleChange("Self_Employed", v)}
                options={[{ value: 0, label: "No" }, { value: 1, label: "Yes" }]} />
              <Toggle label="Dependents" value={input.Dependents} onChange={(v) => handleChange("Dependents", v)}
                options={[{ value: 0, label: "0" }, { value: 1, label: "1" }, { value: 2, label: "2" }, { value: 4, label: "3+" }]} />
              <Toggle label="Property area" value={input.Property_Area} onChange={(v) => handleChange("Property_Area", v)}
                options={[{ value: 0, label: "Rural" }, { value: 1, label: "Semiurban" }, { value: 2, label: "Urban" }]} />
            </div>

            <div style={{
              margin: "16px 0 20px", padding: "14px 16px", borderRadius: 3,
              background: input.Credit_History ? "rgba(94,143,107,0.10)" : "rgba(168,69,59,0.10)",
              border: `1px solid ${input.Credit_History ? "#3E5C46" : "#5C3E38"}`
            }}>
              <Toggle label="Credit history on file — the single largest factor in this model" value={input.Credit_History}
                onChange={(v) => handleChange("Credit_History", v)}
                options={[{ value: 0, label: "No record" }, { value: 1, label: "Clean record" }]} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 28 }}>
              <NumSlider name="ApplicantIncome" value={input.ApplicantIncome} onChange={handleChange} prefix="$" />
              <NumSlider name="CoapplicantIncome" value={input.CoapplicantIncome} onChange={handleChange} prefix="$" />
              <NumSlider name="LoanAmount" value={input.LoanAmount} onChange={handleChange} prefix="$" />
              <NumSlider name="Loan_Amount_Term" value={input.Loan_Amount_Term} onChange={handleChange} prefix="" />
            </div>
          </div>

          <div style={{
            background: "rgba(20,28,22,0.72)", border: "1px solid #263229", borderRadius: 4,
            padding: "30px 24px", display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", backdropFilter: "blur(6px)"
          }}>
            <div style={{
              width: 108, height: 108, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
              border: `3px solid ${approved ? "#5E8F6B" : "#A8453B"}`, marginBottom: 18,
              transition: "border-color 400ms ease"
            }}>
              <div style={{ fontSize: 28, fontFamily: "'Courier New', monospace", color: approved ? "#8FC29A" : "#D68078", fontWeight: 700 }}>
                {Math.round(probability * 100)}%
              </div>
            </div>
            <div style={{
              fontSize: 15, letterSpacing: "0.03em", color: approved ? "#8FC29A" : "#D68078", marginBottom: 8,
              fontFamily: "Georgia, serif"
            }}>
              {approved ? "Reads: approved" : "Reads: declined"}
            </div>
            <div style={{ fontSize: 12.5, color: "#7C8578", textAlign: "center", lineHeight: 1.6, maxWidth: 240 }}>
              {approved
                ? "This profile's combination of signals sits with the model's approved cases."
                : "This profile's combination of signals sits with the model's declined cases."}
            </div>
            <div style={{ marginTop: 20, fontSize: 11, color: "#525A4C", textAlign: "center" }}>
              cohort approval rate: {Math.round(OVERALL_RATE * 100)}%
            </div>
          </div>
        </div>

        {/* Feature contributions */}
        <div style={{ marginBottom: 54 }}>
          <SectionLabel>What is moving this read — signed contribution of each field, this application</SectionLabel>
          <div style={{ background: "rgba(20,28,22,0.55)", border: "1px solid #263229", borderRadius: 4, padding: "20px 24px 8px" }}>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={importanceData} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
                <CartesianGrid stroke="#1A2318" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#5A6352", fontSize: 11 }} axisLine={{ stroke: "#263229" }} tickLine={false} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fill: "#B7BEA9", fontSize: 12.5, fontFamily: "Georgia, serif" }} axisLine={false} tickLine={false} />
                <ReferenceLine x={0} stroke="#33402C" />
                <Tooltip cursor={{ fill: "rgba(184,147,90,0.06)" }} contentStyle={{ background: "#141C16", border: "1px solid #2A3324", borderRadius: 3, fontSize: 12.5 }} labelStyle={{ color: "#E8E4D8" }} formatter={(v) => [v.toFixed(3), "contribution"]} />
                <Bar dataKey="value" radius={2}>
                  {importanceData.map((d, i) => <Cell key={i} fill={d.value >= 0 ? "#5E8F6B" : "#A8453B"} fillOpacity={0.85} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: "#525A4C", padding: "6px 4px 16px" }}>
              Green pushes toward approval, red pulls toward decline — scaled by each field's weight in the trained SVM.
            </div>
          </div>
        </div>

        <div style={{ height: 1, background: "linear-gradient(90deg, transparent, #263229, transparent)", margin: "0 0 54px" }} />

        {/* Cohort context */}
        <div style={{ marginBottom: 20 }}>
          <SectionLabel>How approval breaks down across the {m.n_total}-application ledger</SectionLabel>
        </div>
        <div className="lp-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28, marginBottom: 54 }}>
          <MiniBar title="By credit history" data={creditBarData} />
          <MiniBar title="By property area" data={areaBarData} />
          <MiniBar title="By education" data={eduBarData} />
          <MiniBar title="By marital status" data={marriedBarData} />
        </div>

        <div style={{ height: 1, background: "linear-gradient(90deg, transparent, #263229, transparent)", margin: "0 0 54px" }} />

        {/* Model diagnostics */}
        <div style={{ marginBottom: 20 }}>
          <SectionLabel>How the underlying model performs — held-out test split, {m.n_test} of {m.n_total} applications</SectionLabel>
        </div>
        <div className="lp-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>
          <div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 20 }}>
              <StatCard label="Test accuracy" value={`${Math.round(m.test_accuracy * 100)}%`} sub={`train ${Math.round(m.train_accuracy * 100)}%`} accent="#E9C46A" />
              <StatCard label="Precision" value={m.precision.toFixed(2)} sub="of predicted-approved" />
              <StatCard label="Recall" value={m.recall.toFixed(2)} sub="of actual-approved found" />
              <StatCard label="ROC AUC" value={m.roc_auc.toFixed(2)} />
            </div>
            <div style={{ background: "rgba(20,28,22,0.55)", border: "1px solid #263229", borderRadius: 4, padding: 20 }}>
              <div style={{ fontSize: 12, color: "#7C8578", marginBottom: 14 }}>Confusion matrix</div>
              <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 1, fontSize: 12.5 }}>
                <div />
                <div style={{ textAlign: "center", color: "#5A6352", paddingBottom: 6 }}>pred. declined</div>
                <div style={{ textAlign: "center", color: "#5A6352", paddingBottom: 6 }}>pred. approved</div>
                <div style={{ color: "#5A6352", display: "flex", alignItems: "center", paddingRight: 8 }}>actual declined</div>
                <div style={{ background: "#17251A", textAlign: "center", padding: "14px 0", color: "#8FC29A", fontSize: 18 }}>{cm[0][0]}</div>
                <div style={{ background: "#241814", textAlign: "center", padding: "14px 0", color: "#D68078", fontSize: 18 }}>{cm[0][1]}</div>
                <div style={{ color: "#5A6352", display: "flex", alignItems: "center", paddingRight: 8 }}>actual approved</div>
                <div style={{ background: "#241814", textAlign: "center", padding: "14px 0", color: "#D68078", fontSize: 18 }}>{cm[1][0]}</div>
                <div style={{ background: "#17251A", textAlign: "center", padding: "14px 0", color: "#8FC29A", fontSize: 18 }}>{cm[1][1]}</div>
              </div>
              <div style={{ fontSize: 11, color: "#525A4C", marginTop: 14, lineHeight: 1.5 }}>
                Recall of {m.recall.toFixed(2)} means the model rarely turns away applicants who were actually approved —
                its misses lean toward approving a few who were declined.
              </div>
            </div>
          </div>

          <div style={{ background: "rgba(20,28,22,0.55)", border: "1px solid #263229", borderRadius: 4, padding: "20px 24px" }}>
            <div style={{ fontSize: 12, color: "#7C8578", marginBottom: 14 }}>ROC curve — AUC {m.roc_auc.toFixed(3)}</div>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={ROC_CURVE} margin={{ left: -12, right: 12, top: 4, bottom: 4 }}>
                <CartesianGrid stroke="#1A2318" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fill: "#5A6352", fontSize: 11 }} axisLine={{ stroke: "#263229" }} tickLine={false} label={{ value: "false-positive rate", position: "insideBottom", offset: -4, fill: "#525A4C", fontSize: 11 }} />
                <YAxis dataKey="tpr" type="number" domain={[0, 1]} tick={{ fill: "#5A6352", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#141C16", border: "1px solid #2A3324", borderRadius: 3, fontSize: 12 }} labelStyle={{ color: "#E8E4D8" }} formatter={(v, n) => [v.toFixed(2), n]} />
                <Line type="monotone" dataKey="tpr" stroke="#B8935A" strokeWidth={2} dot={false} />
                <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#33402C" strokeDasharray="3 4" dot={false} strokeWidth={1} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{ marginTop: 60, paddingTop: 20, borderTop: "1px solid #1A2318", fontSize: 11.5, color: "#454C3E", lineHeight: 1.6 }}>
          Trained on {m.n_total} applications from the uploaded dataset ({m.n_approved} approved, {m.n_declined} declined) with
          a linear-kernel SVM over standardized features, following the notebook's encoding exactly. Probabilities are
          Platt-scaled from the model's decision function. This is a demonstration of the notebook's model, not a lending
          decision or financial advice.
        </div>
      </div>
    </div>
  );
}

function MiniBar({ title, data }) {
  return (
    <div>
      <SectionLabel>{title}</SectionLabel>
      <div style={{ background: "rgba(20,28,22,0.55)", border: "1px solid #263229", borderRadius: 4, padding: "16px 20px 6px" }}>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 36, top: 4, bottom: 4 }}>
            <CartesianGrid stroke="#1A2318" horizontal={false} />
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis type="category" dataKey="name" width={110} tick={{ fill: "#B7BEA9", fontSize: 12, fontFamily: "Georgia, serif" }} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "rgba(184,147,90,0.06)" }} contentStyle={{ background: "#141C16", border: "1px solid #2A3324", borderRadius: 3, fontSize: 12 }} labelStyle={{ color: "#E8E4D8" }} formatter={(v) => [`${Math.round(v * 100)}%`, "approval rate"]} />
            <Bar dataKey="value" radius={2} fill="#B8935A" fillOpacity={0.85} label={{ position: "right", formatter: (v) => `${Math.round(v * 100)}%`, fill: "#8A9086", fontSize: 11 }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
