#!/usr/bin/env node

const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const packageJson = require('../package.json');
const VERSION = packageJson.version;
const BINARY_VERSION = VERSION;
const PLATFORM = os.platform();
const ARCH = os.arch();

const PLATFORM_MAP = {
  darwin: 'apple-darwin',
  linux: 'unknown-linux-gnu',
  win32: 'pc-windows-msvc',
};

const ARCH_MAP = {
  x64: 'x86_64',
  arm64: 'aarch64',
};

const platformStr = PLATFORM_MAP[PLATFORM];
const archStr = ARCH_MAP[ARCH];

if (!platformStr || !archStr) {
  console.error(`Unsupported platform: ${PLATFORM}-${ARCH}`);
  console.error('Supported release assets: macOS x64/arm64, Linux x64, Windows x64');
  process.exit(1);
}

const target = `${archStr}-${platformStr}`;
const SUPPORTED_TARGETS = new Set([
  'aarch64-apple-darwin',
  'x86_64-apple-darwin',
  'x86_64-unknown-linux-gnu',
  'x86_64-pc-windows-msvc',
]);
if (!SUPPORTED_TARGETS.has(target)) {
  console.error(`Unsupported Vestige release target: ${target}`);
  console.error('Supported release assets:');
  for (const supported of SUPPORTED_TARGETS) {
    console.error(`  - ${supported}`);
  }
  process.exit(1);
}

const isWindows = PLATFORM === 'win32';
const archiveExt = isWindows ? 'zip' : 'tar.gz';
const archiveName = `vestige-mcp-${target}.${archiveExt}`;
const downloadUrl = `https://github.com/samvallad33/vestige/releases/download/v${BINARY_VERSION}/${archiveName}`;

const targetDir = path.join(__dirname, '..', 'bin');
const archivePath = path.join(targetDir, archiveName);
const checksumPath = path.join(targetDir, `${archiveName}.sha256`);
const expectedArchiveMembers = new Set(
  ['vestige-mcp', 'vestige', 'vestige-restore'].map((name) => (isWindows ? `${name}.exe` : name))
);

function isWorkspaceCheckout() {
  const packageRoot = path.resolve(__dirname, '..');
  const repoRoot = path.resolve(packageRoot, '..', '..');
  return (
    path.basename(packageRoot) === 'vestige-mcp-npm' &&
    path.basename(path.dirname(packageRoot)) === 'packages' &&
    fs.existsSync(path.join(repoRoot, 'pnpm-workspace.yaml'))
  );
}

if (process.env.VESTIGE_SKIP_BINARY_DOWNLOAD === '1' || isWorkspaceCheckout()) {
  console.log('Skipping Vestige binary download in local workspace checkout.');
  process.exit(0);
}

console.log(`Installing Vestige MCP v${VERSION} for ${target}...`);

// Ensure bin directory exists
if (!fs.existsSync(targetDir)) {
  fs.mkdirSync(targetDir, { recursive: true });
}

/**
 * Download a file following redirects (GitHub releases use redirects)
 */
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);

    const request = (currentUrl) => {
      https.get(currentUrl, (response) => {
        // Handle redirects (GitHub uses 302)
        if (response.statusCode === 301 || response.statusCode === 302) {
          const redirectUrl = response.headers.location;
          if (!redirectUrl) {
            reject(new Error('Redirect without location header'));
            return;
          }
          request(redirectUrl);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Download failed: HTTP ${response.statusCode}`));
          return;
        }

        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(dest, () => {}); // Delete partial file
        reject(err);
      });
    };

    request(url);
  });
}

/**
 * Extract archive based on platform
 */
function extract(archivePath, destDir) {
  validateArchiveEntries(archivePath);
  if (isWindows) {
    // Use PowerShell to extract zip on Windows
    execFileSync(
      'powershell',
      [
        '-NoProfile',
        '-Command',
        `Expand-Archive -LiteralPath ${powershellQuote(archivePath)} -DestinationPath ${powershellQuote(destDir)} -Force`,
      ],
      { stdio: 'inherit' }
    );
  } else {
    // Use tar on Unix
    execFileSync('tar', ['-xzf', archivePath, '-C', destDir], { stdio: 'inherit' });
  }
}

function powershellQuote(value) {
  return `'${String(value).replace(/'/g, "''")}'`;
}

function listArchiveEntries(archivePath) {
  if (!isWindows) {
    return execFileSync('tar', ['-tzf', archivePath], { encoding: 'utf8' });
  }

  const script = [
    'Add-Type -AssemblyName System.IO.Compression.FileSystem;',
    `$zip = [System.IO.Compression.ZipFile]::OpenRead(${powershellQuote(archivePath)});`,
    'try { $zip.Entries | ForEach-Object { $_.FullName } } finally { $zip.Dispose() }',
  ].join(' ');
  return execFileSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8' });
}

function normalizeArchiveEntry(entry) {
  let normalized = entry.replace(/\\/g, '/').replace(/^\.\//, '');
  if (
    !normalized ||
    normalized.startsWith('/') ||
    /^[A-Za-z]:/.test(normalized) ||
    normalized.split('/').some((part) => part === '' || part === '..')
  ) {
    throw new Error(`Unsafe archive entry: ${entry}`);
  }
  return normalized;
}

function validateArchiveEntries(archivePath) {
  const entries = listArchiveEntries(archivePath)
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean);

  for (const entry of entries) {
    const normalized = normalizeArchiveEntry(entry);
    if (!expectedArchiveMembers.has(normalized)) {
      throw new Error(`Unexpected archive entry: ${entry}`);
    }
  }
}

/**
 * Make binaries executable (Unix only)
 */
function makeExecutable(binDir) {
  if (isWindows) return;

  const binaries = ['vestige-mcp', 'vestige', 'vestige-restore'];
  for (const bin of binaries) {
    const binPath = path.join(binDir, bin);
    if (fs.existsSync(binPath)) {
      fs.chmodSync(binPath, 0o755);
    }
  }
}

function verifyChecksum(archivePath, checksumPath) {
  const checksumText = fs.readFileSync(checksumPath, 'utf8').trim();
  const expected = checksumText.split(/\s+/)[0]?.toLowerCase();
  if (!expected || !/^[a-f0-9]{64}$/.test(expected)) {
    throw new Error(`Invalid checksum file for ${archiveName}`);
  }

  const actual = crypto.createHash('sha256').update(fs.readFileSync(archivePath)).digest('hex');
  if (actual !== expected) {
    throw new Error(`Checksum mismatch for ${archiveName}`);
  }
}

async function main() {
  try {
    // Download
    console.log(`Downloading from ${downloadUrl}...`);
    await download(downloadUrl, archivePath);
    await download(`${downloadUrl}.sha256`, checksumPath);
    verifyChecksum(archivePath, checksumPath);
    console.log('Download complete.');

    // Extract
    console.log('Extracting binaries...');
    extract(archivePath, targetDir);

    // Cleanup archive
    fs.unlinkSync(archivePath);
    fs.unlinkSync(checksumPath);

    // Make executable
    makeExecutable(targetDir);

    // Verify installation
    const mcpBinary = path.join(targetDir, isWindows ? 'vestige-mcp.exe' : 'vestige-mcp');
    const cliBinary = path.join(targetDir, isWindows ? 'vestige.exe' : 'vestige');

    if (!fs.existsSync(mcpBinary)) {
      throw new Error('vestige-mcp binary not found after extraction');
    }

    console.log('');
    console.log('Vestige MCP installed successfully!');
    console.log('');
    console.log('Binaries installed:');
    console.log(`  - vestige-mcp: ${mcpBinary}`);
    if (fs.existsSync(cliBinary)) {
      console.log(`  - vestige:     ${cliBinary}`);
    }
    console.log('');
    console.log('Next steps:');
    console.log('  1. Add vestige-mcp to any MCP-compatible agent.');
    console.log('     Claude Code: claude mcp add vestige vestige-mcp -s user');
    console.log('     Codex:       codex mcp add vestige -- vestige-mcp');
    console.log('     OpenCode:    npx @vestige/init, or add mcp.vestige to ~/.config/opencode/opencode.json');
    console.log('  2. Restart your MCP client.');
    console.log('  3. Test with: "remember that my preferred editor is VS Code"');
    console.log('');

  } catch (err) {
    console.error('');
    console.error('Installation failed:', err.message);
    console.error('');
    console.error('Manual installation:');
    console.error(`  1. Download: ${downloadUrl}`);
    console.error(`  2. Extract to: ${targetDir}`);
    console.error('  3. Ensure binaries are executable (chmod +x on Unix)');
    console.error('');
    process.exit(1);
  }
}

main();
