const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const album = require('../lib/vestige-album.js');
const share = require('../lib/vestige-share.js');
const { FIXTURES } = require('../lib/vestige-booking.js');

const strip = (text) => text.replace(/\x1b\[[0-9;]*m/g, '');
const OUT = path.join(require('os').tmpdir(), 'vestige-init-tests', 'share', `test-${Date.now()}`);
const now = Date.parse('2026-10-12T03:52:00Z');
const make = (i, commitment, extra = {}) => ({ ...FIXTURES.b, commitment, digest: (i + 1).toString(16).padStart(2, '0').repeat(8), ts: now - i * 3600000, ...extra });
const canon = [{ number: 1, version: 'v4.2.0', title: 'the deletion', rule: 'OP-001' }];

test('rarity comes from local frequency only', () => {
  const stats = { total: 100, byCommitment: { 'OP-001': 60, 'OP-018': 4, 'OP-005': 5 }, firstTs: { 'OP-001': 1, 'OP-018': 2, 'OP-005': 3 } };
  assert.deepEqual(album.rarity({ commitment: 'OP-001', ts: 1 }, stats), { tier: 'first', label: 'first OP-001 here' });
  assert.deepEqual(album.rarity({ commitment: 'OP-001', ts: 9 }, stats), { tier: 'common', label: '60 of 100 here' });
  assert.deepEqual(album.rarity({ commitment: 'OP-018', ts: 9 }, stats), { tier: 'rare', label: '4 of 100 here' });
  assert.equal(album.rarity({ commitment: 'OP-005', ts: 9 }, stats).tier, 'common', 'exactly 5 percent is not rare');
});

test('the album fits its width and hides what is not yet seen', () => {
  const stories = [make(0, 'OP-001', { action: 'SECRETACTION', source: 'zzagent' }), make(1, 'OP-018'), make(2, 'OP-001')];
  const stats = { total: 3, byCommitment: { 'OP-001': 2, 'OP-018': 1 }, firstTs: { 'OP-001': stories[2].ts, 'OP-018': stories[1].ts } };
  const seen = new Set([stories[1].digest, stories[2].digest]);
  for (const cols of [40, 60, 80, 120, 200]) {
    const lines = album.albumView({ stories, canon, seen, stats, cols });
    for (const line of lines) assert.ok([...strip(line)].length <= cols, `${cols}: ${strip(line)}`);
    assert.ok(!/;48;|\[4[0-7]m/.test(lines.join('')), 'no background colours');
    const text = strip(lines.join('\n'));
    assert.ok(text.startsWith(cols >= 60 ? '3 true stories · 1 canon episode · 1 not yet seen' : '3 true stories · 1 canon episode'), 'header');
    assert.ok(text.includes('not yet seen') && text.includes('┄'));
    assert.ok(!text.includes('SECRETACTION') && !text.includes('zzagent') && !text.includes(stories[0].digest.slice(0, 8)));
    assert.ok(text.includes('first OP-018 here') && text.includes('EPISODE 1'));
  }
  assert.equal(strip(album.albumView({ stories, canon, seen, stats, cols: 120 })[2]).split('┌').length - 1, 4, 'four tiles fit at 120 columns');
  const empty = strip(album.albumView({ canon, cols: 80 }).join('\n'));
  assert.ok(empty.includes('0 true stories') && empty.includes('after Operator stops something'));
  const many = Array.from({ length: 30 }, (_v, i) => make(i, 'OP-001'));
  assert.ok(strip(album.albumView({ stories: many, cols: 100 }).join('\n')).includes('and 6 older true stories'));
});

test('artifacts are saved where asked and odd names are refused', () => {
  const env = { VESTIGE_REM_OUT: OUT };
  const file = share.saveArtifact(Buffer.from('x'), 'vestige-rem-abc.png', env);
  assert.equal(file, path.join(OUT, 'vestige-rem-abc.png'));
  assert.equal(fs.readFileSync(file, 'utf8'), 'x');
  for (const name of ['../x.png', 'a/b.png', '.hidden', 'sp ace.png', '']) assert.throws(() => share.saveArtifact(Buffer.from('x'), name, env));
  assert.equal(share.outDir({}, path.join(OUT, 'no-such-home')), path.join(OUT, 'no-such-home'));
  fs.mkdirSync(path.join(OUT, 'home', 'Downloads'), { recursive: true });
  assert.equal(share.outDir({}, path.join(OUT, 'home')), path.join(OUT, 'home', 'Downloads'));
});

test('clipboard commands are built as argument lists, never shell strings', () => {
  const awkward = '/tmp/a "quoted" \\ name/card.png';
  const calls = [];
  const record = (command, args, options) => { calls.push([command, args, options]); };
  assert.deepEqual(share.copyImage(awkward, { platform: 'darwin', run: record }), { ok: true, via: 'osascript' });
  assert.equal(calls[0][0], 'osascript');
  assert.equal(calls[0][1][1], 'set the clipboard to (read (POSIX file "/tmp/a \\"quoted\\" \\\\ name/card.png") as «class PNGf»)');
  share.copyFile(awkward, { platform: 'darwin', run: record });
  assert.equal(calls[1][1][1], 'set the clipboard to (POSIX file "/tmp/a \\"quoted\\" \\\\ name/card.png")');
  share.copyImage(awkward, { platform: 'win32', run: record });
  assert.equal(calls[2][0], 'powershell');
  assert.equal(calls[2][1].at(-1), awkward, 'the path travels as its own argument');
  const failing = (command) => { if (command === 'wl-copy') throw new Error('missing'); calls.push([command]); };
  const real = path.join(OUT, 'vestige-rem-abc.png');
  assert.deepEqual(share.copyImage(real, { platform: 'linux', run: failing }), { ok: true, via: 'xclip' });
  assert.deepEqual(share.copyImage(real, { platform: 'linux', run: () => { throw new Error('none'); } }), { ok: false, via: '' });
  assert.deepEqual(share.copyFile(real, { platform: 'linux', run: record }), { ok: true, via: 'wl-copy' });
  assert.equal(calls.at(-1)[2].input, 'file://' + real.split('/').map(encodeURIComponent).join('/') + '\n');
});

// Security review 2026-10-18, finding 3: an export never writes through a planted link.
test('a planted symlink at the export name is replaced, never written through', (t) => {
  const dir = path.join(OUT, 'symlink');
  fs.mkdirSync(dir, { recursive: true });
  const victim = path.join(dir, 'victim.txt');
  fs.writeFileSync(victim, 'keep me');
  const name = 'vestige-rem-0123abcd.png';
  try {
    fs.symlinkSync(victim, path.join(dir, name));
  } catch (err) {
    if (process.platform === 'win32' && err && err.code === 'EPERM') {
      t.skip('Windows symlink privilege unavailable');
      return;
    }
    throw err;
  }
  const file = share.saveArtifact(Buffer.from('PNGDATA'), name, { VESTIGE_REM_OUT: dir });
  assert.equal(fs.readFileSync(victim, 'utf8'), 'keep me', 'the link target is untouched');
  assert.equal(fs.lstatSync(file).isSymbolicLink(), false, 'the export is a regular file now');
  assert.equal(fs.readFileSync(file, 'utf8'), 'PNGDATA');
  assert.equal(fs.readdirSync(dir).filter((n) => n.endsWith('.part')).length, 0, 'no temp file is left');
  // a second export replaces the first, as before
  share.saveArtifact(Buffer.from('AGAIN'), name, { VESTIGE_REM_OUT: dir });
  assert.equal(fs.readFileSync(file, 'utf8'), 'AGAIN');
});
