const SEP = '<sep>';
const EOS = '<eos>';
const PAD = '<pad>';

let d = 64;
let maxLen = 64;
let epoch = 1000;
let nHead = 4;
let maxtokens = 64;
const lr = 0.003;
const nLayer = 3;

let vocab, V, stoi, itos;
let model;
let step = 1;

function tokenize(s) {
  return s.toLowerCase().trim().split(/\s+/);
}

class Mat {
  constructor(r, c) {
    this.rows = r;
    this.cols = c;
    this.w = new Float32Array(r * c);
    this.dw = new Float32Array(r * c);
    this.m = new Float32Array(r * c);
    this.v = new Float32Array(r * c);
  }
  static rand(r, c) {
    const m = new Mat(r, c);
    const s = 1 / Math.sqrt(c);
    for (let i = 0; i < m.w.length; i++) m.w[i] = (Math.random() * 2 - 1) * s;
    return m;
  }
}

class Graph {
  constructor(bp = true) {
    this.bp = bp;
    this.fns = [];
  }
  backward() {
    for (let i = this.fns.length - 1; i >= 0; i--) this.fns[i]();
  }
  matmul(a, b) {
    const o = new Mat(a.rows, b.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < b.cols; j++) {
        let s = 0;
        for (let k = 0; k < a.cols; k++) s += a.w[i * a.cols + k] * b.w[k * b.cols + j];
        o.w[i * b.cols + j] = s;
      }
    }
    if (this.bp) {
      this.fns.push(() => {
        for (let i = 0; i < a.rows; i++) {
          for (let j = 0; j < b.cols; j++) {
            const g = o.dw[i * b.cols + j];
            if (!g) continue;
            for (let k = 0; k < a.cols; k++) {
              a.dw[i * a.cols + k] += b.w[k * b.cols + j] * g;
              b.dw[k * b.cols + j] += a.w[i * a.cols + k] * g;
            }
          }
        }
      });
    }
    return o;
  }
  add(a, b) {
    const o = new Mat(a.rows, a.cols);
    for (let i = 0; i < o.w.length; i++) o.w[i] = a.w[i] + b.w[i];
    if (this.bp) {
      this.fns.push(() => {
        for (let i = 0; i < o.w.length; i++) {
          a.dw[i] += o.dw[i];
          b.dw[i] += o.dw[i];
        }
      });
    }
    return o;
  }
  tanh(m) {
    const o = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.w.length; i++) o.w[i] = Math.tanh(m.w[i]);
    if (this.bp) {
      this.fns.push(() => {
        for (let i = 0; i < m.w.length; i++) m.dw[i] += (1 - o.w[i] ** 2) * o.dw[i];
      });
    }
    return o;
  }
  embed(m, ix) {
    const o = new Mat(1, m.cols);
    for (let i = 0; i < m.cols; i++) o.w[i] = m.w[ix * m.cols + i];
    if (this.bp) {
      this.fns.push(() => {
        for (let i = 0; i < m.cols; i++) m.dw[ix * m.cols + i] += o.dw[i];
      });
    }
    return o;
  }
  softmax(m) {
    const o = new Mat(m.rows, m.cols);
    for (let i = 0; i < m.rows; i++) {
      let max = -1e9;
      for (let j = 0; j < m.cols; j++) max = Math.max(max, m.w[i * m.cols + j]);
      let sum = 0;
      for (let j = 0; j < m.cols; j++) sum += Math.exp(m.w[i * m.cols + j] - max);
      for (let j = 0; j < m.cols; j++) o.w[i * m.cols + j] = Math.exp(m.w[i * m.cols + j] - max) / sum;
    }
    if (this.bp) {
      this.fns.push(() => {
        for (let i = 0; i < m.rows; i++) {
          for (let j = 0; j < m.cols; j++) {
            const g = o.dw[i * m.cols + j];
            for (let k = 0; k < m.cols; k++) {
              m.dw[i * m.cols + k] += g * o.w[i * m.cols + k] * ((j === k ? 1 : 0) - o.w[i * m.cols + j]);
            }
          }
        }
      });
    }
    return o;
  }
}

function adam(p) {
  const b1 = 0.9, b2 = 0.999, e = 1e-8;
  for (let i = 0; i < p.w.length; i++) {
    p.m[i] = b1 * p.m[i] + (1 - b1) * p.dw[i];
    p.v[i] = b2 * p.v[i] + (1 - b2) * p.dw[i] ** 2;
    const mh = p.m[i] / (1 - Math.pow(b1, step));
    const vh = p.v[i] / (1 - Math.pow(b2, step));
    p.w[i] -= lr * mh / (Math.sqrt(vh) + e);
    p.dw[i] = 0;
  }
}

function forward(ids, g) {
  const n = ids.length;
  const dh = d / nHead;
  let x = new Mat(n, d);
  for (let i = 0; i < n; i++) {
    const e = g.add(g.embed(model.wte, ids[i]), g.embed(model.wpe, i));
    for (let j = 0; j < d; j++) x.w[i * d + j] = e.w[j];
  }
  for (const L of model.layers) {
    const heads = [];
    for (let h = 0; h < nHead; h++) {
      const Q = g.matmul(x, L.wq[h]);
      const K = g.matmul(x, L.wk[h]);
      const Vp = g.matmul(x, L.wv[h]);
      const att = new Mat(n, n);
      const s = 1 / Math.sqrt(d / nHead);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          let v = 0;
          for (let k = 0; k < d / nHead; k++) v += Q.w[i * (d / nHead) + k] * K.w[j * (d / nHead) + k];
          att.w[i * n + j] = v * s;
        }
      }
      heads.push(g.matmul(g.softmax(att), Vp));
    }
    const cat = new Mat(n, d);
    for (let i = 0; i < n; i++)
      for (let h = 0; h < nHead; h++)
        for (let k = 0; k < d / nHead; k++)
          cat.w[i * d + h * (d / nHead) + k] = heads[h].w[i * (d / nHead) + k];
    x = g.add(x, g.matmul(cat, L.wo));
    x = g.add(x, g.matmul(g.tanh(g.matmul(x, L.w1)), L.w2));
  }
  return g.matmul(x, model.wh);
}

function train() {
  traindata = useTraindata;

  const time = Date.now();
  step = 1;
  const dh = d / nHead;

  const words = new Set([PAD, SEP, EOS]);
  traindata.forEach(x => {
    tokenize(x.q).forEach(w => words.add(w));
    tokenize(x.a).forEach(w => words.add(w));
  });
  vocab = Array.from(words);
  V = vocab.length;
  stoi = Object.fromEntries(vocab.map((w, i) => [w, i]));
  itos = Object.fromEntries(vocab.map((w, i) => [i, w]));

  model = {
    wte: Mat.rand(V, d),
    wpe: Mat.rand(maxLen, d),
    layers: Array.from({ length: nLayer }, () => ({
      wq: Array.from({ length: nHead }, () => Mat.rand(d, dh)),
      wk: Array.from({ length: nHead }, () => Mat.rand(d, dh)),
      wv: Array.from({ length: nHead }, () => Mat.rand(d, dh)),
      wo: Mat.rand(d, d),
      w1: Mat.rand(d, d * 4),
      w2: Mat.rand(d * 4, d)
    })),
    wh: Mat.rand(d, V)
  };

  for (let e = 0; e < epoch; e++) {
    for (let i = 0; i < traindata.length; i++) {
      step++;
      const { q, a } = traindata[Math.floor(Math.random() * traindata.length)];
      const seq = [...tokenize(q), SEP, ...tokenize(a), EOS].map(w => stoi[w]);
      if (seq.length < 2) continue;
      const g = new Graph(true);
      const logits = forward(seq.slice(0, -1), g);
      for (let t = 0; t < seq.length - 1; t++) {
        const off = t * V;
        let max = -1e9;
        for (let v = 0; v < V; v++) max = Math.max(max, logits.w[off + v]);
        let sum = 0;
        for (let v = 0; v < V; v++) sum += Math.exp(logits.w[off + v] - max);
        for (let v = 0; v < V; v++) {
          const p = Math.exp(logits.w[off + v] - max) / sum;
          logits.dw[off + v] += p - (v === seq[t + 1] ? 1 : 0);
        }
      }
      g.backward();
      adam(model.wte);
      adam(model.wpe);
      adam(model.wh);
      model.layers.forEach(L => {
        L.wq.forEach(adam);
        L.wk.forEach(adam);
        L.wv.forEach(adam);
        adam(L.wo);
        adam(L.w1);
        adam(L.w2);
      });
    }
  }
  return Date.now() - time;
}

function generate(text, n = maxtokens) {
  let seq = [...tokenize(text), SEP].map(w => stoi[w]).filter(x => x != null);
  const sepId = stoi[SEP], eosId = stoi[EOS];
  for (let i = 0; i < n; i++) {
    const ctx = seq.slice(-maxLen);
    const g = new Graph(false);
    const logits = forward(ctx, g);
    const off = (ctx.length - 1) * V;
    let max = -1e9;
    for (let v = 0; v < V; v++) if (v !== sepId) max = Math.max(max, logits.w[off + v]);
    let sum = 0, probs = [];
    for (let v = 0; v < V; v++) {
      if (v === sepId) { probs[v] = 0; continue; }
      probs[v] = Math.exp(logits.w[off + v] - max);
      sum += probs[v];
    }
    let r = Math.random() * sum, idx = 0;
    for (; idx < V; idx++) { r -= probs[idx]; if (r <= 0) break; }
    if (idx === eosId) break;
    seq.push(idx);
  }
  const w = seq.map(i => itos[i]);
  return w.slice(w.indexOf(SEP) + 1).join(' ');
}

function ask(q) { return generate(q); }

function allmodel() {
  const w = {
    wte: Array.from(model.wte.w),
    wpe: Array.from(model.wpe.w),
    wh: Array.from(model.wh.w),
    layers: model.layers.map(L => ({
      wq: L.wq.map(mat => Array.from(mat.w)),
      wk: L.wk.map(mat => Array.from(mat.w)),
      wv: L.wv.map(mat => Array.from(mat.w)),
      wo: Array.from(L.wo.w),
      w1: Array.from(L.w1.w),
      w2: Array.from(L.w2.w)
    }))
  };
  return { vocab, stoi, itos, w };
}