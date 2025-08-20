export default {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: [
    '<rootDir>/tests/**/*.test.{ts,tsx}',
    '<rootDir>/src/**/*.test.{ts,tsx}',
    '<rootDir>/tests/**/*.spec.{ts,tsx}',
    '<rootDir>/src/**/*.spec.{ts,tsx}'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
    '^.+\\.(js|jsx)$': 'babel-jest'
  },
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@voice/(.*)$': '<rootDir>/src/voice/$1',
    '^@visualization/(.*)$': '<rootDir>/src/visualization/$1',
    '^@ai/(.*)$': '<rootDir>/src/ai/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1'
  },
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.ts',
    '<rootDir>/tests/setup-audio.ts',
    '<rootDir>/tests/setup-webgl.ts'
  ],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.{ts,tsx}',
    '!src/**/*.test.{ts,tsx}',
    '!src/**/*.spec.{ts,tsx}',
    '!src/index.tsx',
    '!src/vite-env.d.ts'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'lcov',
    'html',
    'json',
    'json-summary'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/coverage/'
  ],
  moduleFileExtensions: [
    'ts',
    'tsx',
    'js',
    'jsx',
    'json',
    'node'
  ],
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
      useESM: true
    }
  },
  extensionsToTreatAsEsm: ['.ts', '.tsx'],
  testTimeout: 30000,
  maxWorkers: '50%',
  verbose: true,
  detectOpenHandles: true,
  forceExit: true
};
