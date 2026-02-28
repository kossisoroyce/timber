import type {SidebarsConfig} from '@docusaurus/types';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    'how-it-works',
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/loading-models',
        'guides/serving-models',
        'guides/python-predictor',
        'guides/embedding-in-c',
        'guides/wasm-deployment',
        'guides/misra-c-compliance',
        'guides/differential-compilation',
        'guides/ensemble-composition',
        'guides/audit-trails',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/xgboost',
        'examples/lightgbm',
        'examples/sklearn',
        'examples/catboost',
        'examples/onnx',
        'examples/http-client',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api-reference/cli',
        'api-reference/http-api',
        'api-reference/python-api',
        'api-reference/c-api',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/compiler-pipeline',
        'architecture/intermediate-representation',
        'architecture/optimization-passes',
        'architecture/code-generation',
      ],
    },
    'configuration',
    'troubleshooting',
    'contributing',
  ],
};

export default sidebars;
