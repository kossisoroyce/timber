import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';

function HeroSection() {
  return (
    <div className="hero--timber">
      <div className="container">
        <h1 className="hero__title">Timber</h1>
        <p className="hero__subtitle">
          Compile XGBoost, LightGBM, scikit-learn, CatBoost &amp; ONNX models to native C99.
          Serve any model — local file or remote URL — in one command.
        </p>

        <div style={{margin: '2rem auto 1.5rem', maxWidth: 480}}>
          <code style={{
            display: 'block',
            background: '#1a1a1a',
            color: '#ffffff',
            padding: '0.75rem 1.25rem',
            borderRadius: 4,
            fontSize: '1rem',
            fontFamily: 'monospace',
            border: '1px solid #333',
            textAlign: 'left',
          }}>
            pip install timber-compiler
          </code>
        </div>

        <div style={{
          background: '#111',
          border: '1px solid #2a2a2a',
          borderRadius: 6,
          padding: '1.1rem 1.5rem',
          maxWidth: 680,
          margin: '0 auto 2rem',
          textAlign: 'left',
          fontFamily: 'monospace',
          fontSize: '0.95rem',
          lineHeight: 1.9,
          color: '#aaaaaa',
        }}>
          <div style={{color: '#555', marginBottom: '0.25rem', fontSize: '0.8rem'}}>then run:</div>
          <div>
            <span style={{color: '#555'}}>$ </span>
            <span style={{color: '#ffffff'}}>timber serve https://yourhost.com/model.json</span>
          </div>
          <div style={{color: '#444', fontSize: '0.82rem', marginTop: '0.4rem'}}>
            ↳ downloads · compiles · serves — all in one command
          </div>
        </div>

        <div style={{display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap'}}>
          <Link className="button button--primary button--lg" to="/docs/getting-started">
            Get Started →
          </Link>
          <Link className="button button--outline button--lg" style={{color: '#aaaaaa', borderColor: '#444'}} to="https://github.com/kossisoroyce/timber">
            GitHub ★
          </Link>
        </div>
      </div>
    </div>
  );
}

function StatsSection() {
  return (
    <div className="stats-bar">
      <div className="stat-item">
        <span className="stat-value">336×</span>
        <span className="stat-label">Faster P50 Latency</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">533×</span>
        <span className="stat-label">Higher Throughput</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">91µs</span>
        <span className="stat-label">HTTP Inference</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">48KB</span>
        <span className="stat-label">Compiled Artifact</span>
      </div>
      <div className="stat-item">
        <span className="stat-value">5</span>
        <span className="stat-label">Frameworks</span>
      </div>
    </div>
  );
}

function QuickstartSection() {
  const codeUrl = `# Option 1: serve directly from a URL — no pre-download needed
pip install timber-compiler
timber serve https://yourhost.com/fraud_model.json`;

  const codeLocal = `# Option 2: load a local file, then serve by name
pip install timber-compiler
timber load fraud_model.json --name fraud-detector
timber serve fraud-detector

# Query either way
curl http://localhost:11434/api/predict \\
  -d '{"model": "fraud-detector", "inputs": [[1.0, 2.0, ...]]}'`;

  return (
    <div className="container" style={{padding: '3rem 0'}}>
      <div className="row">
        <div className="col col--5" style={{display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
          <h2 style={{fontSize: '2rem', lineHeight: 1.2}}>One command.<br />Native speed.</h2>
          <p style={{color: 'var(--ifm-color-secondary-darkest)', fontSize: '1rem', lineHeight: 1.7, marginTop: '1rem'}}>
            Point Timber at any URL and it downloads, compiles, and serves immediately.
            No separate load step. No configuration. Python never touches the hot path.
          </p>
          <p style={{fontSize: '0.9rem', color: '#888'}}>
            Or load a local file and serve by name — your choice.
          </p>
        </div>
        <div className="col col--7">
          <CodeBlock language="bash" title="Serve from URL (recommended)">
            {codeUrl}
          </CodeBlock>
          <div style={{marginTop: '1rem'}}>
            <CodeBlock language="bash" title="Serve from local file">
              {codeLocal}
            </CodeBlock>
          </div>
        </div>
      </div>
    </div>
  );
}

function FeaturesSection() {
  const features = [
    {
      title: '5 Framework Parsers',
      description: 'XGBoost, LightGBM, scikit-learn, CatBoost, and ONNX. Auto-detected from file extension and content.',
    },
    {
      title: '6 Optimizer Passes',
      description: 'Dead leaf elimination, constant folding, threshold quantization, branch sorting, pipeline fusion, vectorization analysis.',
    },
    {
      title: '3 Code Backends',
      description: 'C99 for servers & embedded, WebAssembly for browsers & edge, MISRA-C for safety-critical (automotive, medical).',
    },
    {
      title: 'Ollama-Style Serving',
      description: 'timber load → timber serve. REST API on port 11434. Same developer experience as Ollama, but for classical ML.',
    },
    {
      title: 'Zero Dependencies',
      description: 'Generated code needs only a C99 compiler. No runtime libraries, no dynamic allocation, no recursion. Thread-safe by design.',
    },
    {
      title: 'Audit Trails',
      description: 'Every compilation produces a deterministic JSON audit report with SHA-256 hashes, pass logs, and timing. Built for regulated industries.',
    },
  ];

  return (
    <div className="container" style={{padding: '3rem 0'}}>
      <h2 style={{textAlign: 'center', fontSize: '2rem', marginBottom: '2rem'}}>Why Timber?</h2>
      <div className="row">
        {features.map((f, i) => (
          <div key={i} className="col col--4" style={{marginBottom: '1.5rem'}}>
            <div className="feature-card">
              <h3>{f.title}</h3>
              <p style={{color: '#888', margin: 0}}>{f.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ArchitectureSection() {
  return (
    <div className="container" style={{padding: '3rem 0'}}>
      <h2 style={{textAlign: 'center', fontSize: '2rem', marginBottom: '1rem'}}>How It Works</h2>
      <p style={{textAlign: 'center', color: '#888', marginBottom: '2rem', maxWidth: 600, margin: '0 auto 2rem'}}>
        Timber treats your trained model as a program specification and compiles it through a classical compiler pipeline.
      </p>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexWrap: 'wrap',
        gap: '0.5rem',
        fontSize: '0.9rem',
        padding: '1.5rem',
        background: 'var(--ifm-background-surface-color)',
        borderRadius: 4,
        border: '1px solid rgba(128,128,128,0.15)',
      }}>
        {[
          'Model Artifact',
          '→ Parse →',
          'Timber IR',
          '→ Optimize →',
          'Optimized IR',
          '→ Emit →',
          'C99 / WASM',
          '→ Compile →',
          '.so / .dylib',
          '→ Serve →',
          'HTTP API',
        ].map((step, i) => (
          <span key={i} style={{
            padding: step.startsWith('→') ? '0.25rem 0' : '0.4rem 0.9rem',
            background: step.startsWith('→') ? 'transparent' : 'rgba(128,128,128,0.1)',
            borderRadius: step.startsWith('→') ? 0 : 4,
            color: step.startsWith('→') ? '#888' : 'var(--ifm-font-color-base)',
            fontWeight: step.startsWith('→') ? 400 : 600,
            fontFamily: 'monospace',
          }}>
            {step}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Home" description={siteConfig.tagline}>
      <HeroSection />
      <main>
        <StatsSection />
        <QuickstartSection />
        <FeaturesSection />
        <ArchitectureSection />
        <div style={{textAlign: 'center', padding: '3rem 0'}}>
          <Link className="button button--primary button--lg" to="/docs/getting-started">
            Read the Docs →
          </Link>
        </div>
      </main>
    </Layout>
  );
}
