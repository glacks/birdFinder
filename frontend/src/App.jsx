import { useMemo, useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const topLabel = useMemo(() => {
    if (!result?.top_prediction) return "-";
    const name = result.top_prediction.class_name;
    const prob = (result.top_prediction.probability * 100).toFixed(2);
    return `${name} (${prob}%)`;
  }, [result]);

  const onChangeFile = (event) => {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResult(null);
    setError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : "");
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("이미지를 먼저 선택해주세요.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `요청 실패 (${res.status})`);
      }

      const body = await res.json();
      setResult(body);
    } catch (e) {
      setError(e.message || "예측 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="bg-shape bg-shape-left" />
      <div className="bg-shape bg-shape-right" />

      <main className="card">
        <h1>Bird Finder</h1>
        <p className="sub">사진을 올리면 새 종을 예측합니다.</p>

        <form onSubmit={onSubmit} className="form">
          <label className="filebox">
            <span>{file ? file.name : "이미지 선택 (jpg, png)"}</span>
            <input type="file" accept="image/*" onChange={onChangeFile} />
          </label>
          <button disabled={loading} type="submit">
            {loading ? "분석 중..." : "분류 시작"}
          </button>
        </form>

        <section className="result">
          <h2>Top-1</h2>
          <p className="top1">{topLabel}</p>

          {previewUrl && (
            <div className="preview-wrap">
              <img src={previewUrl} alt="preview" className="preview" />
            </div>
          )}

          {error && <p className="error">{error}</p>}

          {result?.top_k?.length > 0 && (
            <ul className="rank">
              {result.top_k.map((item) => (
                <li key={item.class_name}>
                  <span>{item.class_name}</span>
                  <strong>{(item.probability * 100).toFixed(2)}%</strong>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
