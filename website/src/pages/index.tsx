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
          Ollama for classical ML models. Compile XGBoost, LightGBM, scikit-learn,
          CatBoost & ONNX models into native C99 inference code.
          One command to load, one command to serve.
        </p>
        <div style={{marginTop: '2rem', display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap'}}>
          <Link className="button button--primary button--lg" to="/docs/getting-started">
            Get Started →
          </Link>
          <Link className="button button--outline button--lg" style={{color: '#94a3b8', borderColor: '#334155'}} to="https://github.com/kossisoroyce/timber">
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
  const code = `# Install
pip install timber-compiler

# Load & compile a model
timber load model.json --name fraud-detector

# Serve it
timber serve fraud-detector

# Query it
curl http://localhost:11434/api/predict \\
  -d '{"model": "fraud-detector", "inputs": [[1.0, 2.0, ...]]}'`;

  return (
    <div className="container" style={{padding: '3rem 0'}}>
      <div className="row">
        <div className="col col--5" style={{display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
          <h2 style={{fontSize: '2rem'}}>Three commands. Native speed.</h2>
          <p style={{color: '#94a3b8', fontSize: '1.1rem', lineHeight: 1.7}}>
            Timber compiles your trained model to optimized C99, caches the shared library locally,
            and serves it over an Ollama-compatible REST API. Python never touches the inference hot path.
          </p>
        </div>
        <div className="col col--7">
          <CodeBlock language="bash" title="Terminal">
            {code}
          </CodeBlock>
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
              <p style={{color: '#94a3b8'}}>{f.description}</p>
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
      <p style={{textAlign: 'center', color: '#94a3b8', marginBottom: '2rem', maxWidth: 600, margin: '0 auto 2rem'}}>
        Timber treats your trained model as a program specification and compiles it through a classical compiler pipeline.
      </p>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexWrap: 'wrap',
        gap: '0.5rem',
        fontSize: '0.95rem',
        padding: '1.5rem',
        background: 'var(--ifm-background-surface-color)',
        borderRadius: 12,
        border: '1px solid rgba(52,211,153,0.1)',
      }}>
        {[
          'Model Artifact',
          '→ Parse →',
          'Timber IR',
          '→ Optimize (6 passes) →',
          'Optimized IR',
          '→ Emit →',
          'C99 / WASM',
          '→ Compile →',
          '.so / .dylib',
          '→ Serve →',
          'HTTP API',
        ].map((step, i) => (
          <span key={i} style={{
            padding: step.startsWith('→') ? '0.25rem 0' : '0.5rem 1rem',
            background: step.startsWith('→') ? 'transparent' : 'rgba(52,211,153,0.1)',
            borderRadius: step.startsWith('→') ? 0 : 8,
            color: step.startsWith('→') ? '#64748b' : 'var(--ifm-color-primary)',
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
