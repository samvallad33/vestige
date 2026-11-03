#!/usr/bin/env node
'use strict';
// Local, fail-open lifecycle client. No model/provider calls; only explicit user statements
// and agent-submitted facts are admitted. Memory text is always presented as untrusted data.
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');
const readline = require('readline');
const { defaults, trigger } = require('../lib/memory-triggers.cjs');
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const atomic = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value), { mode: 0o600 });
  fs.renameSync(temp, file);
};
function rpc(binary, args = [], timeoutMs = 210, env = process.env) {
  const child = spawn(binary, args, { stdio: ['pipe', 'pipe', 'pipe'], env });
  let diagnostics = '';
  child.stderr.on('data', chunk => { if (env.VESTIGE_HOOK_DIAGNOSTICS === '1') diagnostics = (diagnostics + chunk).slice(-2000); });
  const pending = new Map(); let seq = 0; let closed = false; let disposed = false; let timer;
  const rejectAll = error => { closed = true; for (const [,p] of pending) p.reject(error); pending.clear(); };
  child.on('error', rejectAll);
  child.on('exit', () => rejectAll(new Error('Vestige disconnected')));
  child.stdin.on('error', rejectAll);
  const lines = readline.createInterface({ input: child.stdout });
  lines.on('line', line => {
    let msg; try { msg = JSON.parse(line); } catch { return; }
    const p = pending.get(msg.id); if (!p) return;
    pending.delete(msg.id);
    if (msg.error) p.reject(new Error(msg.error.message || 'RPC failed')); else p.resolve(msg.result);
  });
  const call = (method, params) => new Promise((resolve, reject) => {
    if (closed) return reject(new Error('Vestige unavailable'));
    const id = ++seq; pending.set(id, { resolve, reject, method });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
  const close = () => {
    if (disposed) return;
    disposed = true; clearTimeout(timer);
    rejectAll(new Error('Vestige transport closed'));
    // The result is already admitted, or its queue entry remains for retry. A slow
    // child shutdown must not keep the chat hook alive beyond its work budget.
    lines.close();
    child.stdin.destroy(); child.stdout.destroy(); child.stderr.destroy();
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    child.unref();
  };
  timer = setTimeout(() => { const stages = [...pending.values()].map(p => p.method).join(','); rejectAll(new Error(`memory deadline exceeded (${stages})${diagnostics ? `: ${diagnostics}` : ''}`)); close(); }, timeoutMs);
  return {
    diagnostics() { return diagnostics; },
    async tool(name, arguments_) {
      const r = await call('tools/call', { name, arguments: arguments_ });
      if (r.isError) throw new Error(`Vestige ${name} refused`);
      return r.structuredContent || JSON.parse(r.content[0].text);
    },
    async ready() {
      await call('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'vestige-memory-hook', version: '6.2.0' } });
      child.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n');
    },
    close,
  };
}

const INDEX_PREFIXES = ['facts-of:', 'day:', 'yearly:', 'avoid:'];
let factEngine;
function loadFactEngine() {
  if (factEngine) return factEngine;
  // Deterministic text helpers ported from salience-pro (STARA) companion.rs:
  // COMMON, IRREGULAR, GIVEN_NAMES, WORD_NAMES, NICKNAMES, fold, tokens, forms.
  // No sound(), no edit distance. Identity stays exact.
  const COMMON = ["a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so", "some", "that", "the", "their", "them", "then", "they", "this", "to", "was", "we", "well", "were", "what", "when", "who", "why", "with", "you", "ok", "okay", "yeah", "yes", "nope", "hey", "hi", "hello", "like", "really", "again", "still", "now", "though", "please", "one"];
  const GIVEN_NAMES = ["aaron", "abigail", "adam", "aiden", "alice", "amanda", "amelia", "amy", "andrew", "angela", "anna", "anthony", "ashley", "ava", "barbara", "benjamin", "betty", "brandon", "brian", "carol", "caroline", "charles", "charlotte", "chloe", "christopher", "daniel", "david", "deborah", "donna", "dorothy", "dylan", "edward", "elijah", "elizabeth", "ella", "ellie", "emily", "emma", "eric", "ethan", "evelyn", "gabriel", "grace", "hannah", "harper", "helen", "henry", "isaac", "isabella", "jack", "jacob", "james", "jason", "jennifer", "jessica", "john", "jonathan", "joseph", "joshua", "julia", "justin", "karen", "kevin", "kimberly", "laura", "leo", "liam", "lily", "linda", "logan", "lucas", "lucy", "luke", "madison", "maria", "mark", "mary", "mason", "matthew", "maya", "megan", "melissa", "mia", "michael", "michelle", "nancy", "natalie", "nathan", "nicole", "noah", "olivia", "oliver", "patricia", "paul", "rachel", "rebecca", "richard", "robert", "ryan", "samantha", "samuel", "sandra", "sara", "sarah", "scarlett", "sophia", "stephanie", "steven", "susan", "thomas", "tyler", "victoria", "william", "zoe"];
  const WORD_NAMES = ["will", "may", "june", "april", "august", "rose", "grace", "hope", "faith", "joy", "bill", "rob", "pat", "art", "mark", "drew", "chase", "sunny", "rich", "frank", "jack", "max", "summer", "autumn", "dawn", "skye", "river", "storm", "ray", "rocky", "harmony", "page"];
  const NICKNAMES = [["abby", "abigail"], ["alex", "alexander|alexandra"], ["andy", "andrew"], ["ben", "benjamin"], ["beth", "elizabeth"], ["bill", "william"], ["bob", "robert"], ["cathy", "catherine|katherine"], ["charlie", "charles|charlotte"], ["chris", "christopher|christina|christine"], ["dan", "daniel"], ["danny", "daniel"], ["dave", "david"], ["ellie", "eleanor|elizabeth|elena"], ["jen", "jennifer"], ["jenny", "jennifer"], ["jim", "james"], ["jimmy", "james"], ["joe", "joseph"], ["josh", "joshua"], ["kate", "katherine|catherine|kaitlyn"], ["katie", "katherine|catherine|kaitlyn"], ["liz", "elizabeth"], ["lizzie", "elizabeth"], ["maddie", "madison|madeline"], ["maddy", "madison|madeline"], ["matt", "matthew"], ["meg", "megan|margaret"], ["maggie", "margaret"], ["peggy", "margaret"], ["mike", "michael"], ["mikey", "michael"], ["nick", "nicholas"], ["rob", "robert"], ["sam", "samuel|samantha"], ["steve", "steven|stephen"], ["tom", "thomas"], ["tommy", "thomas"], ["tony", "anthony"], ["will", "william"], ["zach", "zachary"], ["izzy", "isabella|isabel"], ["bella", "isabella"], ["sophie", "sophia"]];
  const IRREGULAR = [["children", "child"], ["people", "person"], ["mice", "mouse"], ["teeth", "tooth"], ["feet", "foot"], ["men", "man"], ["women", "woman"], ["geese", "goose"], ["ran", "run"], ["swam", "swim"], ["won", "win"], ["went", "go"], ["ate", "eat"], ["bought", "buy"], ["made", "make"], ["wrote", "write"], ["sang", "sing"], ["drew", "draw"], ["flew", "fly"], ["rode", "ride"], ["taught", "teach"], ["brought", "bring"], ["allergic", "allergy"], ["fought", "fight"], ["angry", "mad"]];
  const NICK = new Map();
  for (const [short, full] of NICKNAMES) for (const name of full.split('|')) { NICK.set(short, name); if (!NICK.has(name)) NICK.set(name, name); }
  const SITUATION_WORDS = {
    animal: 'dog puppy cat kitten pet horse rabbit bird hamster fish turtle vet adopt shelter',
    child: 'kid child son daughter baby toddler school daycare playground',
    'food-meal': 'cook bake dinner lunch breakfast recipe meal restaurant dish pizza pasta soup',
    'drink-alcohol': 'wine beer cocktail bar pub brewery whiskey vodka',
    'medicine-taking': 'medication medicine pill painkiller supplement vitamin prescription antibiotic inhaler insulin',
    'medical-care': 'doctor hospital surgery appointment checkup therapy therapist dentist clinic diagnosis injection vaccine',
    exertion: 'run marathon hike climb gym workout train yoga cycle bike lift',
    water: 'swim surf beach lake boat kayak sail dive pool',
    outdoors: 'garden park camp forest nature mountain trail flower plant',
    romance: 'date boyfriend girlfriend wedding anniversary engage marry propose valentine',
    'celebrate-success': 'graduate promotion award achievement celebrate',
    'family-day': 'thanksgiving christmas birthday mother father holiday reunion',
    'baby-news': 'pregnant pregnancy newborn baby',
    party: 'party concert festival gathering club',
    'big-spend': 'buy bought house car apartment mortgage purchase',
    trip: 'trip vacation travel journey road visit',
    'travel-abroad': 'abroad overseas country passport international',
    driving: 'drive car road truck',
    flight: 'flight fly airport plane',
    'job-change': 'job hire promote quit career interview offer resign position internship',
    health: 'sick ill injure hurt pain flu cold fever allergy allergic asthma diabetes cancer anxiety depression recover',
    hobby: 'paint draw sing dance guitar piano music read book write photograph knit game volunteer',
    education: 'class course study exam degree college university semester school',
  };
  const SIT_INDEX = new Map();
  for (const [key, words] of Object.entries(SITUATION_WORDS)) for (const word of words.split(' ')) SIT_INDEX.set(word, key);

  function fold(text) {
    const map = { 'ß': 'ss', '\u2018': "'", '\u2019': "'", '\u02BC': "'", '`': "'" };
    return String(text).replace(/[\u200B-\u200D\uFEFF\u00AD]/g, '').replace(/[ß\u2018\u2019\u02BC`]/g, ch => map[ch])
      .replace(/[🎂🎉🥳]/gu, ' birthday ').replace(/[🐶🐕🐩]/gu, ' dog ').replace(/[🐱🐈]/gu, ' cat ').replace(/🍕/gu, ' pizza ')
      .replace(/🎮/gu, ' game ').replace(/🏀/gu, ' basketball ').replace(/⚽/gu, ' soccer ').replace(/[🎸🎵🎶🎤]/gu, ' music ')
      .replace(/[📚📖]/gu, ' book ').replace(/🏫/gu, ' school ')
      .toLowerCase().replace(/ł/g, 'l').replace(/ø/g, 'o').replace(/đ/g, 'd').normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  }
  function tokens(text) {
    const out = [];
    for (const raw of fold(text).split(/[^\p{L}\p{N}]+/u)) {
      if (!raw) continue;
      if (/^\d+(?:st|nd|rd|th)$/.test(raw)) { out.push(raw); continue; }
      out.push(...(raw.match(/\p{N}+|\p{L}+/gu) || []));
    }
    return out.filter(word => word !== 's');
  }
  function forms(word) {
    const out = [word];
    const n = [...word].length;
    for (const [a, b] of IRREGULAR) { if (word === a) out.push(b); else if (word === b) out.push(a); }
    if (!/^[\x00-\x7f]*$/.test(word) || n < 4 || COMMON.includes(word)) return [...new Set(out)].sort();
    const cut = k => word.slice(0, word.length - k);
    const undouble = base => (base.length > 2 && base.at(-1) === base.at(-2) && 'bdgmnprt'.includes(base.at(-1))) ? base.slice(0, -1) : null;
    if (word.endsWith('s') && !word.endsWith('ss')) {
      out.push(cut(1));
      if (word.endsWith('es')) out.push(cut(2));
      if (word.endsWith('ies') && n > 4) out.push(cut(3) + 'y');
      if (word.endsWith('ves') && n > 4) { out.push(cut(3) + 'f'); out.push(cut(3) + 'fe'); }
    }
    for (const [end, keep] of [['ing', 3], ['ed', 2]]) {
      if (word.endsWith(end) && n > keep + 2) {
        const base = cut(keep);
        out.push(base + 'e');
        const single = undouble(base);
        if (single) out.push(single);
        if (end === 'ed' && base.endsWith('i')) out.push(base.slice(0, -1) + 'y');
        out.push(base);
      }
    }
    return [...new Set(out)].sort();
  }
  function knownName(word, atStart) {
    const folded = fold(word);
    if (WORD_NAMES.includes(folded)) return !atStart && /^\p{Lu}/u.test(word);
    return GIVEN_NAMES.includes(folded) || NICK.has(folded);
  }
  function situations(text) {
    const hit = new Set();
    for (const token of tokens(text)) for (const form of forms(token)) if (SIT_INDEX.has(form)) hit.add(SIT_INDEX.get(form));
    return [...hit];
  }
  function phraseKeys(value) {
    let words = tokens(value);
    if (words.some(word => !COMMON.includes(word))) words = words.filter(word => !COMMON.includes(word));
    if (!words.length) return [];
    const headForms = forms(words[words.length - 1]);
    return [words.join(''), ...headForms.map(head => [...words.slice(0, -1), head].join(''))];
  }
  function reuseSpelling(value, known) {
    const keys = new Set(phraseKeys(value));
    for (const item of known || []) if (phraseKeys(item).some(key => keys.has(key))) return item;
    return value;
  }

  const SELF = /^(?:(?:Now|Currently|These days|Nowadays),?\s+)?(?:I|I'm|I've|I'd|I'll|We|We're|We've|We'll)\b/;
  const YOU = /^(?:You|You're|You've|You'll|Your)\b/;
  const MY = /^(?:My|Our)\s+((?:[a-z]+\s+){0,2}?(?:wife|husband|partner|boyfriend|girlfriend|fianc[eé]e?|son|daughter|kids?|children|mom|mother|dad|father|parents|brother|sister|siblings?|grandma|grandmother|grandpa|grandfather|aunt|uncle|cousin|friend|best friend|boss|coworker|colleague|roommate|neighbou?r|dog|cat|pet))\b/;
  const PRONOUN = /^(?:He|She|They)\b/;
  const NAME = /^([\p{Lu}][\p{Ll}\p{M}'-]+(?:\s+[\p{Lu}][\p{Ll}\p{M}'-]+)?)\b/u;
  const KIN = 'wife|husband|partner|boyfriend|girlfriend|fianc[eé]e?|son|daughter|sons|daughters|kids?|children|child|baby|twins|mom|mother|dad|father|parents|brother|sister|brothers|sisters|siblings?|grandma|grandmother|grandpa|grandfather|grandchildren|grandkids|aunt|uncle|cousin|nephew|niece|dog|cat|puppy|kitten|pet';
  const RULES = [
    ['location', true, /^(?:(?:now|currently|still|recently|just)\s+)?(?:live[sd]?|living|reside[sd]?|moved|relocated|am moving|are moving|is moving|have moved|has moved|'ve moved|'m moving|'re moving)\b.*?\b(?:in|to|into)\s+(?:the\s+)?([\p{Lu}][\p{L}\p{M}'-]*(?:\s+[\p{Lu}][\p{L}\p{M}'-]*){0,3})/u],
    ['origin', true, /^(?:am|is|are|'m|'re|was|were)\s+(?:originally\s+)?from\s+([\p{Lu}][\p{L}\p{M}'-]*(?:\s+[\p{Lu}][\p{L}\p{M}'-]*){0,3})/u],
    ['job', true, /^(?:(?:now|currently|still|recently|just)\s+)?(?:work(?:s|ed|ing)?|am working|is working|'m working|started working|got a job|got hired|was hired|switched (?:jobs?|careers?)|took a job|joined)\b(?:\s+(?:as|at|for|in|with)\s+|\s+)(.{2,80})/u],
    ['job', true, /^(?:am|is|'m|are|'re)\s+(?:now\s+)?(?:a|an)\s+((?:[a-z-]+\s+){0,2}(?:teacher|nurse|doctor|engineer|developer|programmer|designer|lawyer|manager|student|counsel(?:l)?or|therapist|artist|writer|chef|cook|accountant|scientist|researcher|consultant|driver|pilot|officer|firefighter|mechanic|electrician|plumber|carpenter|farmer|musician|singer|photographer|journalist|professor|lecturer|analyst|architect|pharmacist|dentist|veterinarian|vet|paramedic|soldier|volunteer|intern|freelancer|entrepreneur|owner|founder|retiree|coach|trainer|social worker|librarian|cashier|barista|waiter|waitress|salesperson|realtor|agent|editor|actor|actress|dancer|athlete))\b/u],
    ['relationship', true, /^(?:am|is|are|'m|'re|got|just got|recently got|have been|has been|'ve been)\s+(?:now\s+)?(married|engaged|divorced|separated|single|widowed|dating\s+[\p{Lu}][\p{L}'-]*)/u],
    ['relationship', true, /^(?:married|divorced|broke up with|split up with|started dating|am dating|is dating|got engaged to)\s+(.{2,60})/u],
    ['family', false, new RegExp(`^(?:have|has|'ve got|had|adopted|welcomed)\\s+(?:a|an|two|three|four|five|\\d+|new|baby|little|twin|\\s)*\\s*((?:[a-z-]+\\s+){0,2}(?:${KIN}))\\b`, 'u')],
    ['possession', false, /^(?:(?:just|recently|finally)\s+)?(?:bought|own|owns|purchased|got|'ve got|have got|has got|have|has)\s+(?:a|an|my|our|his|her|their|the|new|two|three|\d+)\s+((?:new\s+|used\s+|old\s+|little\s+|big\s+)?(?:[a-z-]+\s+){0,2}(?:car|truck|bike|motorcycle|house|home|apartment|condo|boat|laptop|computer|phone|guitar|piano|camera|tv|console|watch|ring|necklace|painting|tattoo|garden|business|shop|store|restaurant|cabin|farm|horse|tent|kayak|telescope|instrument|collection))\b/u],
    ['plan', false, /^(?:(?:am|is|are|'m|'re)\s+(?:planning|going|hoping|about|scheduled|set)\s+to|plan(?:s|ning)? to|will|'ll|intend(?:s)? to|want(?:s)? to|hope(?:s)? to|(?:am|is|are|'m|'re)\s+(?:starting|leaving|flying|traveling|travelling|moving|getting|having|hosting|attending|visiting|running|taking))\s+(.{3,120})/u],
    ['preference', false, /^(?:(?:really|absolutely|still|now)\s+)?(?:like[sd]?|love[sd]?|hate[sd]?|enjoy(?:s|ed)?|prefer(?:s|red)?|adore[sd]?|dislike[sd]?|can't stand|am into|'m into|is into|are into|am allergic to|'m allergic to|is allergic to|am a (?:big )?fan of|'m a (?:big )?fan of)\s+(.{2,100})/u],
    ['preference', false, /^(?:no longer|don't|do not|doesn't|does not|never)\s+(?:like|love|enjoy|eat|drink|play|watch|go to|want)\s+(.{2,100})/u],
    ['avoid', false, /^(?:can't have|cannot have|can't eat|cannot eat|avoids?|must avoid|stays? away from)\s+(?:any\s+|all\s+)?(.{2,80})/u],
    ['diet', true, /^(?:am|'m|is|are|'re|went|have gone|'ve gone|became|have become|'ve become|been)\s+(?:a\s+|now\s+|fully\s+|strictly\s+)*(vegetarian|vegan|pescatarian|gluten[- ]free|dairy[- ]free|keto|paleo|kosher|halal|sober|teetotal)\b/u],
    ['health', false, /^(?:(?:felt|feels|feel|was feeling|is feeling|am feeling|'m feeling)\s+(off|sick|ill|unwell|nauseous|dizzy|awful|terrible)\b.{0,80}|(?:has|have|had)\s+(?:a |an )?([a-z]+)\s+allergy|(?:am|'m|is|was|were|got|have been|has been|'ve been)\s+(?:still\s+|now\s+|really\s+)?(sick|ill|injured|hurt|pregnant|exhausted|burned out|recovering\b.{0,60}|diagnosed with .{2,60}|in (?:the )?hospital.{0,40})|(?:broke|hurt|injured|sprained|twisted)\s+(?:my|his|her|their)\s+([a-z]+)|(?:have|has|had)\s+(?:a |an )?((?:[a-z]+\s+)?(?:surgery|injury|cold|flu|fever|migraine|condition|diabetes|asthma|anxiety|depression|cancer|appointment)))/u],
    ['hobby', false, /^(?:(?:am|'m|is|are|'re)\s+(?:learning|taking up|practicing|practising|training for|getting into)\s+(.{2,80})|(?:play|plays|paint|paints|draw|draws|swim|swims|knit|knits|sing|sings|dance|dances|hike|hikes|surf|surfs|skate|skates|garden|gardens|practice|practices|collect|collects|volunteer|volunteers|run|runs|bake|bakes|cook|cooks|write|writes|read|reads)\b\s*(.{0,80}))/u],
    ['event', false, /^(?:(?:just|recently|finally|actually)\s+)?(?:went|visited|attended|traveled|travelled|took|ran|finished|completed|graduated|started|joined|adopted|met|won|lost|celebrated|volunteered|signed up|gave|hosted|organized|organised|participated|competed|performed|painted|wrote|published|launched|opened|passed|bought|sold|built|made|learned|learnt|tried|saw|watched|read|climbed|hiked|camped|cooked|baked|planted|moved)\b\s*(.{3,140})/u],
  ];
  const CHANGE = /\b(?:now|moved|switched|changed|no longer|not anymore|anymore|used to|these days|recently became|instead)\b/i;
  const NEGATED = /^(?:no longer|don't|do not|doesn't|does not|never|stopped|quit)\b/i;
  const PERSON_ONLY = /^(?:felt|feels|has (?:a |an )?[a-z]+ allergy|lives|lived|moved|relocated|works|worked|is working|got (?:married|engaged|divorced|a job|hired)|married|divorced|adopted|was born|is from|is originally from|has (?:a |an |two |three |\d+ )?(?:new )?(?:baby|son|daughter|kids?|children|wife|husband|partner|boyfriend|girlfriend|brother|sister|dog|cat)|is (?:married|engaged|divorced|single|pregnant|dating)|(?:now |still |really )?(?:likes|loves|hates|prefers|enjoys)|is allergic|can't have|cannot have|can't eat|avoids|went|visited|attended|graduated|joined|started|quit|bought|owns|plans|is planning|is going to|wants to|celebrated|volunteers|volunteered|paints|painted|runs|ran|plays|played)\b/;
  const STOP_CAPS = new Set(['I','The','A','An','This','That','These','Those','It','There','Here','What','When','Where','Why','How','Who','Which','But','And','Or','So','If','Then','Yes','No','Oh','Hey','Hi','Hello','Thanks','Thank','Wow','Well','Also','Just','Today','Tomorrow','Yesterday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','January','February','March','April','May','June','July','August','September','October','November','December','My','Our','His','Her','Their','We','He','She','They','You','Your','Me','Let','Sure','Okay','OK','Great','Good','Nice','Awesome','Cool','Lol','Haha','Absolutely','Definitely','Totally','Really','Maybe','Anyway','Btw','Success','Hope','Love','Time','Life','Work','Hard','Next']);
  const OPENER = /^(?:(?:yeah|yes|yep|yup|no|nope|oh|ah|wow|hey|hi|hello|well|so|ok(?:ay)?|sure|definitely|totally|absolutely|honestly|actually|anyway|btw|by the way|also|plus|overall|lol|haha|thanks|thank you|congrats|cool|nice|great|awesome|true|right|agreed|exactly|recently|lately|yesterday|today)(?:\s+[\p{Lu}][\p{L}'-]+)?\s*[,!:;.-]+\s*)+/iu;
  const CLAUSE = /\s*(?:[,;:]\s*(?:so|and|but|because|cause|while|since|plus|though|although|and then)?|\s-\s|\s+(?:so|and|but|because|cause|while|since|plus|though|although|and then))\s+(?=(?:I|I'm|I've|I'd|I'll|we|We|we're|we've|You|You're)\s)/u;
  const NONFACT = /\b(?:look into it|haven'?t decided|have not decided|not sure yet|haven'?t chosen|get back to you)\b|^(?:next |this )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|week|weekend)\s+works\b/i;
  const CHAT = /^(?:'ll|will|am going to|'m going to)?\s*(?:definitely\s+|totally\s+|surely\s+)?(?:hope so|think so|gotta\b|(?:got|have|need|gotta) (?:to )?go(?: now)?[.! ]*$|must go|keep (?:you|that|it|this)|check (?:it|that|them) out|let you know|give (?:it|that|them|these) a (?:try|shot|go)|make sure to|take your advice|be waiting|talk (?:to you )?(?:soon|later)|see you|catch up)\b/i;
  const VAGUE = /^(?:(?:a|an|the|my|our|his|her|their|it|this|that|these|those|them|him|there|here|one|some|so|too|much|lot|lots|a lot|stuff|things?|something|anything|everything|nothing|fun|out|up|back|over|well|better|way|it)\s*)+$/i;
  const CONDITION = /\b(asthma|diabetes|arthritis|epilepsy|cancer|anxiety|depression|adhd|autism|ptsd|insomnia|migraines?|high blood pressure|celiac|eczema|scoliosis)\b/i;
  const DIET = /\b(?:am|'m|is|are|'re|went|going|became|become|been)\s+(?:a\s+|now\s+|fully\s+|strictly\s+)*(vegetarian|vegan|pescatarian|gluten[- ]free|dairy[- ]free|keto|paleo|kosher|halal|sober|teetotal)\b/i;
  const MEDICATION = /\b(?:taking|take|takes|started|on|prescribed)\s+(?:a\s+|my\s+|some\s+|new\s+)*([a-z-]+\s+)?(medication|medicine|pills?|antibiotics?|antidepressants?|insulin|inhaler|ibuprofen|aspirin|statins?|vitamins?|supplements?)\b/i;
  const slug = value => value.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '').slice(0, 60);
  const head = value => {
    const words = value.replace(/[.,;:!?"')(].*$/, '').replace(/^(?:as|at|for|in|with|to|a|an|the)\s+/i, '').replace(/^(?:a|an|the)\s+/i, '').trim().split(/\s+/).filter(Boolean);
    return words.slice(-3).join(' ');
  };

  function sentences(prompt) {
    const out = [];
    const prose = String(prompt).replace(/```[\s\S]*?```/g, ' ').replace(/\r/g, '');
    for (let text of prose.split(/\n+|(?<=[.!?])\s+(?=[\p{Lu}"'(])/u)) {
      text = text.trim().replace(/^[-*•>\d.)\s]+/u, '');
      const parts = text.split(CLAUSE).map(part => part.trim()).filter(Boolean);
      for (let part of parts) {
        part = part.replace(OPENER, '').trim();
        if (/^[a-z]/.test(part) && /^(?:i|we|my)\b/.test(part)) part = part[0].toUpperCase() + part.slice(1);
        out.push(part);
      }
    }
    return out.filter(text => text.length >= 8 && text.length <= 400);
  }
  function expand(match) {
    if (/'m$|'re$/.test(match)) return 'am ';
    if (/'ve$/.test(match)) return 'have ';
    if (/'ll$/.test(match)) return 'will ';
    if (/'d$/.test(match)) return 'would ';
    return '';
  }
  function subjectOf(sentence, lastPerson, atStart, role) {
    let match;
    if ((match = YOU.exec(sentence))) {
      const who = role === 'assistant' ? 'self' : 'listener';
      const rest = (/^You$/.test(match[0]) ? '' : expand(match[0].slice(3) ? match[0] : '')) + sentence.slice(match[0].length).trim();
      const expanded = /'re$/.test(match[0]) ? 'are ' + sentence.slice(match[0].length).trim()
        : /'ve$/.test(match[0]) ? 'have ' + sentence.slice(match[0].length).trim()
        : /'ll$/.test(match[0]) ? 'will ' + sentence.slice(match[0].length).trim()
        : sentence.slice(match[0].length).trim();
      return { who, rest: expanded || rest };
    }
    if ((match = SELF.exec(sentence))) {
      const who = role === 'assistant' ? 'assistant' : 'self';
      return { who, rest: expand(match[0]) + sentence.slice(match[0].length).trim() };
    }
    if (/^(?:just |finally |recently )?(?:loved|went|visited|got|had|finished|started|tried|bought|moved|joined|took|made|met|saw|spent|signed up|picked up|adopted|graduated|won)\b/i.test(sentence)) {
      return { who: role === 'assistant' ? 'assistant' : 'self', rest: sentence.replace(/^(?:just |finally |recently )/i, '').replace(/^./, ch => ch.toLowerCase()) };
    }
    if ((match = MY.exec(sentence))) return { who: `${role === 'assistant' ? 'assistant' : 'self'}/${slug(match[1])}`, rest: sentence.slice(match[0].length).trim() };
    if ((match = PRONOUN.exec(sentence))) return lastPerson ? { who: lastPerson, rest: sentence.slice(match[0].length).trim(), pronoun: true } : null;
    if ((match = NAME.exec(sentence)) && !STOP_CAPS.has(match[1].split(/\s+/)[0])) {
      const rest = sentence.slice(match[0].length).trim();
      const first = match[1].split(/\s+/)[0];
      if (COMMON.includes(fold(first))) return null;
      if (!(knownName(first, atStart) && /^(?:is|was|has|had|and|'s|will|just|got|[a-z]+(?:s|ed))\b/.test(rest)) && !PERSON_ONLY.test(rest)) return null;
      return { who: match[1], rest, named: true };
    }
    return null;
  }
  function typedAttributes(slot, object, sentence, rest, negated, known) {
    const clean = value => reuseSpelling(String(value).trim().replace(/[.,;:!]+$/, ''), known);
    const out = [];
    const low = rest.toLowerCase();
    if (slot === 'location') out.push(`preference:residence=${clean(object)}`);
    else if (slot === 'origin') out.push(`preference:hometown=${clean(object)}`);
    else if (slot === 'job') {
      const at = /\b(?:at|for|with|joined)\s+([\p{Lu}][\p{L}\p{M}&'-]*(?:\s+[\p{Lu}][\p{L}\p{M}&'-]*){0,3})/u.exec(sentence);
      if (/^(?:quit|left|retired|lost)\b/.test(low)) out.push('preference:employer=none');
      else if (at) out.push(`preference:employer=${clean(at[1])}`);
      const role = /\bas\s+(?:a|an)\s+([a-z -]{2,40}?)(?=[.,;!]|\s+(?:at|for|in|and|since)\b|$)/.exec(rest) || (/^(?:am|is|are|'m)\b/.test(low) ? [null, object] : null);
      if (role && role[1]) out.push(`preference:occupation=${clean(role[1].replace(/^(?:a|an)\s+/, ''))}`);
    } else if (slot === 'relationship') {
      const status = /(married|engaged|divorced|separated|single|widowed|dating|broke up|split up)/i.exec(rest);
      if (status) out.push(`preference:relationship_status=${status[1].toLowerCase().replace(/ /g, '_')}`);
    } else if (slot === 'preference') {
      const thing = clean(object.replace(/^(?:a|an|the|my|our|to|doing|going)\s+/i, ''));
      if (/allergic/i.test(rest)) out.push(`allergy:${thing.toLowerCase()}`);
      else out.push(`preference:${thing.toLowerCase()}=${negated || /^(?:hate|dislike|can't stand)/i.test(rest.replace(/^(?:really|absolutely|still|now)\s+/i, '')) ? 'dislike' : 'like'}`);
    } else if (slot === 'health') {
      if (/\ballergy\b/i.test(rest)) out.push(`allergy:${clean(object).toLowerCase()}`);
      else if (object) out.push(`health:${clean(object).toLowerCase()}`);
    } else if (slot === 'avoid') { /* indexed as avoid:<thing> */ } else if (object) out.push(`${slot}:${clean(object).toLowerCase()}`);
    const condition = CONDITION.exec(sentence);
    if (condition && !negated) out.push(`condition:${condition[1].toLowerCase()}`);
    const diet = DIET.exec(sentence);
    if (diet) {
      const value = diet[1].toLowerCase().replace(/ /g, '-');
      out.push(`diet:${value}`, `preference:diet=${value}`);
    }
    const medication = MEDICATION.exec(sentence);
    if (medication) out.push(`medication:${((medication[1] || '') + medication[2]).trim().toLowerCase()}`);
    return [...new Set(out)];
  }

  // salience-pro 0.3.7 index tags. Deterministic: exact words, no model.
  // facts-of:<who|me> holds every fact about someone, day:YYYY-MM-DD the day a
  // memory falls on, yearly:MM-DD the date it comes round each year, and
  // avoid:<thing> what a fact rules out. They are not identities, so the
  // one-person: rule Backfill needs is untouched.
  const HEALTH_PREFIXES = ['allergy:', 'health:', 'condition:', 'medication:'];
  const MONTHS = ['january','february','march','april','may','june','july','august','september','october','november','december'];
  function thingWords(raw) {
    return String(raw).split(/[^\p{L}\p{N}]+/u).filter(Boolean).map(word => {
      const w = word.toLowerCase(); const n = [...w].length;
      if (n > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
      if (n > 4 && w.endsWith('oes')) return w.slice(0, -2);
      if (n > 3 && w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('us')) return w.slice(0, -1);
      return w;
    });
  }
  function thingKey(raw) { const key = thingWords(raw).join('-'); return [...key].length >= 3 ? key : null; }
  function thingKeys(raw) {
    const words = thingWords(raw); const keys = [];
    for (let len = 1; len <= 3; len++) for (let i = 0; i + len <= words.length; i++) {
      const key = words.slice(i, i + len).join('-');
      if ([...key].length >= 3 && !keys.includes(key)) keys.push(key);
    }
    return keys;
  }
  const pad = n => String(n).padStart(2, '0');
  function dayOfText(raw) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(raw).trim());
    return m && +m[2] >= 1 && +m[2] <= 12 && +m[3] >= 1 && +m[3] <= 31 ? m[0] : null;
  }
  function yearlyOfText(raw) {
    const m = /^(\d{2})-(\d{2})$/.exec(String(raw).trim());
    return m && +m[1] >= 1 && +m[1] <= 12 && +m[2] >= 1 && +m[2] <= 31 ? m[0] : null;
  }
  function datesIn(sentence) {
    const days = []; const yearly = [];
    for (const m of sentence.matchAll(/\b(\d{4}-\d{2}-\d{2})\b/g)) { const d = dayOfText(m[1]); if (d) { days.push(d); yearly.push(d.slice(5)); } }
    const re = new RegExp(`\\b(${MONTHS.join('|')})\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b(?:,?\\s+(\\d{4}))?`, 'gi');
    for (const m of sentence.matchAll(re)) {
      const md = yearlyOfText(`${pad(MONTHS.indexOf(m[1].toLowerCase()) + 1)}-${pad(m[2])}`);
      if (!md) continue; yearly.push(md); if (m[3]) days.push(`${m[3]}-${md}`);
    }
    return { days: [...new Set(days)], yearly: [...new Set(yearly)] };
  }
  const ANNUAL = /\b(birthday|anniversary|died|passed away|death|born)\b/i;
  const AVOID_RULE = /\b(?:avoid|avoids|can't have|cannot have|can't eat|cannot eat|must not eat|shouldn't eat|stays? away from)\s+(?:any\s+|all\s+)?([a-z][a-z -]{1,60}?)(?=[.,;!]|\s+(?:and|but|because|since)\b|$)/i;
  const ENDED = /\b(?:outgrew|outgrown|no longer|not anymore|anymore|isn't allergic|is not allergic|not allergic|never was|was never|wasn't actually|recovered from|got over)\b/i;
  function indexTags(who, sentence, attributes) {
    const tags = [];
    if (ENDED.test(sentence)) return tags;
    const owner = /^(?:self|listener)$/.test(who) ? 'me' : who;
    if (attributes.length) tags.push(`facts-of:${owner}`);
    for (const tag of attributes) if (tag.startsWith('allergy:')) { const key = thingKey(tag.slice(8)); if (key) tags.push(`avoid:${key}`); }
    const avoid = AVOID_RULE.exec(sentence);
    if (avoid) { const key = thingKey(avoid[1]); if (key) tags.push(`avoid:${key}`); }
    if (tags.some(tag => tag.startsWith('avoid:')) && !tags.some(tag => tag.startsWith('facts-of:'))) tags.unshift(`facts-of:${owner}`);
    const { days, yearly } = datesIn(sentence);
    for (const d of days) tags.push(`day:${d}`);
    if (ANNUAL.test(sentence)) for (const d of yearly) tags.push(`yearly:${d}`);
    return [...new Set(tags)];
  }
  const DATED = new RegExp(`^([\\p{Lu}][\\p{Ll}\\p{M}-]+)'s\\s+(birthday|anniversary)\\s+is\\s+(?:on\\s+)?((?:${MONTHS.join('|')})\\s+\\d{1,2}(?:st|nd|rd|th)?|\\d{4}-\\d{2}-\\d{2})`, 'iu');
  function extractFacts(prompt, ctx = {}, options = {}) {
    if (typeof prompt !== 'string' || !prompt.trim() || prompt.length > 20000) return [];
    const role = options.role === 'assistant' ? 'assistant' : 'user';
    const facts = [];
    let lastPerson = ctx.lastPerson || null;
    let speaker = null;
    const known = ctx.known || [];
    const quoted = /^([\p{Lu}][\p{L}\p{M}'-]+(?:\s+[\p{Lu}][\p{L}\p{M}'-]+)?)\s+(?:said|says|wrote|writes|told me|asked)[,:]?\s*["\u201c]([\s\S]*?)["\u201d]?\s*$/u.exec(prompt.trim());
    if (quoted && !STOP_CAPS.has(quoted[1].split(/\s+/)[0])) { speaker = quoted[1]; prompt = quoted[2]; }
    let first = true;
    for (const sentence of sentences(prompt)) {
      const atStart = first;
      first = false;
      if (NONFACT.test(sentence)) continue;
      const people = sentence.split(/\s+/).map(word => word.replace(/[^\p{L}'-]/gu, '').replace(/'s$/, '')).filter((word, index) => word && /^\p{Lu}/u.test(word) && knownName(word, atStart && index === 0) && !STOP_CAPS.has(word));
      if (/\?\s*$/.test(sentence)) { if (people[0]) lastPerson = people[people.length - 1]; continue; }
      if (/^(?:if|when|whether|what if|imagine|suppose|let's|please|can you|could you|would you|do you|did you|are you|is it)\b/i.test(sentence)) continue;
      const dated = DATED.exec(sentence);
      if (dated && !STOP_CAPS.has(dated[1]) && !/^(?:my|our)$/i.test(dated[1])) {
        const md = datesIn(dated[3]).yearly[0] || (dayOfText(dated[3]) || '').slice(5);
        if (md) {
          lastPerson = dated[1];
          const attributes = [`preference:${dated[2].toLowerCase()}=${md}`];
          facts.push({ content: speaker ? `${speaker} said: ${sentence}` : sentence, entities: [`person:${dated[1]}`], attributes, node_type: 'fact',
            source: role === 'assistant' ? 'memory-hook:verbatim-assistant-statement' : 'memory-hook:verbatim-user-statement',
            tags: ['memory-hook-utterance', 'slot:date', `slot:${slug(dated[1])}:${dated[2].toLowerCase()}`, ...indexTags(dated[1], `${dated[2]} ${dated[3]}`, attributes)],
            slot: 'date', slotKey: `slot:${slug(dated[1])}:${dated[2].toLowerCase()}`, single: true, change: false, negated: false, speaker: speaker || dated[1] });
          continue;
        }
      }
      const subject = subjectOf(sentence, lastPerson, atStart, role);
      if (!subject) { if (people[0]) lastPerson = people[people.length - 1]; continue; }
      const ownerName = speaker || (role === 'assistant' && subject.who === 'assistant' ? 'assistant' : subject.who.startsWith('self') ? 'self' : subject.who);
      const who = subject.who.startsWith('self') ? subject.who.replace(/^self/, speaker || 'self') : subject.who === 'assistant' ? (speaker || 'assistant') : subject.who;
      if (subject.named) lastPerson = subject.who;
      else if (people[0] && !subject.pronoun) lastPerson = people[people.length - 1];
      const rest = subject.rest.replace(/^(?:also|really|just|finally|actually|still|recently|totally|definitely)\s+/i, '');
      if (CHAT.test(rest) || VAGUE.test(rest)) continue;
      const content = speaker ? `${speaker} said: ${sentence}` : sentence;
      const place = /\b(?:in|to|at|from|near)\s+([\p{Lu}][\p{L}\p{M}'-]*(?:\s+[\p{Lu}][\p{L}\p{M}'-]*){0,3})/u.exec(sentence);
      const entities = [`person:${who}`];
      const change = CHANGE.test(sentence);
      const negated = NEGATED.test(rest);
      let hit = null;
      for (const [slot, single, rule] of RULES) {
        const match = rule.exec(rest);
        if (!match) continue;
        const object = head((match[1] || match[2] || match[3] || '')).trim();
        if (!object || object.length < 2 || VAGUE.test(object)) continue;
        if ((slot === 'plan') && CHAT.test(match[0])) continue;
        hit = { slot, single, object };
        break;
      }
      if (!hit && (MEDICATION.test(rest) || CONDITION.test(sentence) || DIET.test(sentence))) {
        hit = { slot: 'health', single: false, object: '' };
      }
      if (!hit) continue;
      if (place && hit.slot !== 'job' && !STOP_CAPS.has(place[1].split(/\s+/)[0]) && !knownName(place[1].split(/\s+/)[0], false)) entities.push(`place:${place[1]}`);
      const attributes = ENDED.test(sentence) ? typedAttributes(hit.slot, hit.object, sentence, rest, negated, known).filter(tag => !HEALTH_PREFIXES.some(p => tag.startsWith(p)) && !tag.startsWith('diet:')).slice(0, 32) : typedAttributes(hit.slot, hit.object, sentence, rest, negated, known).slice(0, 32);
      const keyed = attributes.filter(tag => /^preference:[^=]+=/.test(tag)).map(tag => tag.split('=')[0]);
      const slotKey = (hit.single || change || negated) ? `slot:${slug(who)}:${keyed[0] ? slug(keyed[0].slice('preference:'.length)) : hit.slot}` : null;
      for (const tag of attributes) {
        const value = tag.includes('=') ? tag.split('=').slice(1).join('=') : tag.split(':').slice(1).join(':');
        if (value && !known.includes(value)) known.push(value);
      }
      const situationTags = situations(sentence).map(name => `situation:${name}`);
      facts.push({
        content, entities: [...new Set(entities)], attributes, node_type: hit.slot === 'event' || hit.slot === 'plan' ? 'event' : 'fact',
        source: role === 'assistant' ? 'memory-hook:verbatim-assistant-statement' : 'memory-hook:verbatim-user-statement',
        tags: ['memory-hook-utterance', `slot:${hit.slot}`, ...(slotKey ? [slotKey] : []), ...situationTags, ...indexTags(who, sentence, attributes)].slice(0, 40),
        slot: hit.slot, slotKey, single: !!hit.single, change, negated, speaker: ownerName,
      });
    }
    ctx.lastPerson = lastPerson;
    ctx.known = known.slice(-500);
    return facts.slice(0, 20);
  }


  factEngine = { thingWords, thingKey, thingKeys, dayOfText, yearlyOfText, indexTags, extractFacts, sentences };
  return factEngine;
}
function thingWords(...args) { return loadFactEngine().thingWords(...args); }
function thingKey(...args) { return loadFactEngine().thingKey(...args); }
function thingKeys(...args) { return loadFactEngine().thingKeys(...args); }
function dayOfText(...args) { return loadFactEngine().dayOfText(...args); }
function yearlyOfText(...args) { return loadFactEngine().yearlyOfText(...args); }
function indexTags(...args) { return loadFactEngine().indexTags(...args); }
function extractFacts(...args) { return loadFactEngine().extractFacts(...args); }
function sentences(...args) { return loadFactEngine().sentences(...args); }

// A deliberately small declaration grammar. This records the exact utterance and the
// subject it literally names, not an inferred diagnosis, intent, or restaurant ingredient.
function explicitFact(prompt) {
  if (typeof prompt !== 'string' || prompt.length > 4000 || /[?\n]/.test(prompt)) return null;
  const person = /^([\p{Lu}][\p{L}\p{M}'-]{0,79}(?: [\p{Lu}][\p{L}\p{M}'-]{0,79})?) (?:has|is|felt|prefers|likes|loves|hates|works|lives|moved)\b/u.exec(prompt);
  const self = /^I (?:like|prefer|hate|am allergic to)\b/i.test(prompt);
  if (!person && !self) return null;
  const entities = [`person:${person ? person[1] : 'self'}`]; const attributes = [];
  const place = /\bat ([\p{Lu}][\p{L}\p{M}'-]*(?: [\p{Lu}][\p{L}\p{M}'-]*){0,5})(?:[.!]|$)/u.exec(prompt);
  if (place) entities.push(`place:${place[1]}`);
  const allergy = /\bhas (?:a |an )?([\p{L}-]+) allergy\b/iu.exec(prompt);
  if (allergy) attributes.push(`allergy:${allergy[1].toLowerCase()}`);
  return { content: prompt, entities, attributes, node_type: 'event', source: 'memory-hook:verbatim-user-statement', tags: ['memory-hook-utterance'] };
}
function isExactMatch(row) {
  const tags = (row.tags || []).concat(row.entities || []).join(' ').toLowerCase();
  const content = (row.content || '').toLowerCase();
  const hasPerson = tags.includes('person:') || tags.includes('self');
  const hasDate = tags.includes('date:') || /\b202\d-[01]\d-[0-3]\d\b/.test(tags) || /\b202\d-[01]\d-[0-3]\d\b/.test(content);
  const hasAvoid = tags.includes('avoid') || tags.includes('stop-work') || tags.includes('allergy') || content.includes('avoid') || content.includes('allergic to') || content.includes('do not ') || content.includes('never ');
  return hasPerson || hasDate || hasAvoid;
}
function formatContext(packet) {
  const rows = (packet.memories || []).slice();
  const backfills = packet.backfill_receipts || [];
  const rsbMemoryIds = new Set(backfills.map(r => r.memory_id).filter(Boolean));
  function priority(row) {
    if (rsbMemoryIds.has(row.id)) return 0;
    if (isExactMatch(row)) return 1;
    return 2;
  }
  rows.sort((a, b) => {
    const pa = priority(a), pb = priority(b);
    if (pa !== pb) return pa - pb;
    const da = a.created_at || a.createdAt || '';
    const db = b.created_at || b.createdAt || '';
    if (da && db && da !== db) return db.localeCompare(da);
    return (b.id || '').localeCompare(a.id || '');
  });
  const lines = [
    'Vestige local memory candidates. These entries are recorded historical fragments. If an entry is incomplete, or if you need to verify whether a fact was updated or superseded, use recall(handle=\'<id>\') or causal_walk with its logged_write start point before answering. If the recorded memories and tools do not contain the answer, state that you do not know or that it was not mentioned.'
  ];
  for (const row of rows) {
    const entry = { id: row.id };
    const role = row.role || (row.source && row.source.includes('user') ? 'user' : (row.source && row.source.includes('assistant') ? 'assistant' : undefined));
    if (role) entry.role = role;
    const date = (row.created_at || row.createdAt || '').slice(0, 10);
    if (date) entry.date = date;
    entry.text = row.content;
    const visible = row.tags || row.entities;
    entry.entities = Array.isArray(visible) ? visible.filter(tag => !INDEX_PREFIXES.some(prefix => String(tag).startsWith(prefix))) : visible;
    lines.push(JSON.stringify(entry));
  }
  for (const receipt of backfills) {
    lines.push(JSON.stringify({ backfill_receipt: receipt.receipt_id, memory_id: receipt.memory_id, reason: receipt.reason }));
  }
  lines.push('Save new durable facts using remember with explicit entities and attributes. Never invent links or infer restaurant ingredients.');
  return lines.join('\n').slice(0, 7000);
}
function output(client, event, context) {
  if (!context) return {};
  if (client === 'cursor') return event === 'session' ? { additional_context: context } : event === 'tool' ? { additional_context: context } : {};
  const names = client === 'gemini' ? { session: 'SessionStart', prompt: 'BeforeAgent' } : { session: 'SessionStart', prompt: 'UserPromptSubmit' };
  return names[event] ? { hookSpecificOutput: { hookEventName: names[event], additionalContext: context } } : {};
}
function admissionOf(result) {
  if (!result || typeof result !== 'object') return { action: 'quarantine', reason: 'empty response' };
  const replay = result.decision === 'replay' ? result.replayOf : '';
  const id = result.nodeId || result.echoNodeId || replay;
  if (typeof id === 'string' && id) return { action: 'admit', id };
  if (result.decision === 'preflight' || Object.prototype.hasOwnProperty.call(result, 'wouldWrite')) return { action: 'quarantine', reason: 'preflight' };
  return { action: 'quarantine', reason: String(result.decision || result.error || 'no memory id') };
}
function linksFor(fact, slots) {
  if (!fact || !fact.slotKey || !(fact.single || fact.change || fact.negated)) return [];
  const previous = slots && slots[fact.slotKey];
  if (!previous || !previous.id || previous.content === fact.content) return [];
  return [{ kind: 'supersedes', to: previous.id }];
}
function isDeadline(error) {
  const message = error && error.message ? error.message : '';
  return Boolean(error && error.deadline) || /deadline exceeded|Vestige unavailable|disconnected|transport closed/i.test(message);
}
async function drainQueue(facts, ingest) {
  const saved = [];
  const quarantined = [];
  const deferred = [];
  for (let index = 0; index < facts.length; index++) {
    let result;
    try { result = await ingest(facts[index].fact); }
    catch (error) {
      if (isDeadline(error)) { deferred.push(...facts.slice(index)); break; }
      quarantined.push({ file: facts[index].file, reason: (error && error.message) || 'refused' });
      continue;
    }
    const verdict = admissionOf(result);
    if (verdict.action === 'admit') saved.push({ file: facts[index].file, id: verdict.id, fact: facts[index].fact });
    else quarantined.push({ file: facts[index].file, reason: verdict.reason });
  }
  return { saved, quarantined, deferred };
}
function ingestItem(fact) {
  const { queued_at, slot, slotKey, single, change, negated, speaker, file, ...item } = fact;
  return item;
}

async function run({ event, client, configFile, input, deadlineMs = 190 }) {
  const started = performance.now(); let transport;
  const dir = path.dirname(configFile);
  const log = message => { try { const file = path.join(dir, 'errors.log'); if (fs.existsSync(file) && fs.statSync(file).size > 128 * 1024) fs.renameSync(file, `${file}.previous`); fs.appendFileSync(file, `${new Date().toISOString()} ${event}: ${message}\n`, { mode: 0o600 }); } catch {} };
  try {
    const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
    const session = input.session_id || input.conversation_id;
    if (!session || typeof session !== 'string') throw new Error('missing session identifier');
    const queue = path.join(dir, 'pending', digest(`${client}:${session}`));
    const backfillQueue = path.join(dir, 'backfill-pending', digest(`${client}:${session}`));
    const prompt = input.prompt || '';
    const phrases = JSON.parse(fs.readFileSync(config.triggers, 'utf8'));
    if (event === 'prompt') {
      const fact = explicitFact(prompt);
      if (fact) {
        const intent_id = `hook:${digest(`${client}:${session}:${prompt}`)}`;
        const file = path.join(queue, digest(intent_id) + '.json');
        if (!fs.existsSync(file)) atomic(file, { ...fact, intent_id, queued_at: Date.now() });
      }
    }
    transport = rpc(config.binary, config.binaryArgs || [], Math.max(1, deadlineMs - (performance.now() - started)), { ...process.env, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' });
    await transport.ready();
    let saved = [];
    if (['turn', 'compact', 'end'].includes(event)) {
      let facts = [];
      try { facts = fs.readdirSync(queue).filter(f => f.endsWith('.json')).map(file => ({ file:path.join(queue,file), fact:JSON.parse(fs.readFileSync(path.join(queue,file),'utf8')) })).sort((a,b) => a.fact.queued_at-b.fact.queued_at || a.file.localeCompare(b.file)); } catch {}
      // Rich facts are saved by the agent via the MCP tool, not mined from assistant prose.
      for (const { file, fact } of facts.slice(0, 20)) {
        const { queued_at, ...item } = fact;
        const result = await transport.tool('remember', item);
        const admittedId = result.nodeId || result.echoNodeId;
        if (!admittedId || result.success === false) throw new Error('statement admission failed');
        saved.push(admittedId);
        atomic(path.join(backfillQueue, digest(admittedId)+'.json'), { trigger_id:admittedId });
        try { fs.unlinkSync(file); } catch {}
      }
      // Every admitted save already autofires. Revisit these exact triggers on end/compact;
      // pair idempotency makes retries safe without an unbounded full-store sweep.
      if (event !== 'turn') {
        const status = await transport.tool('session', { action:'start', memory_hook:{limit:1} });
        if (status.backfill_autofire === true) {
          let pending=[]; try { pending=fs.readdirSync(backfillQueue).filter(f=>f.endsWith('.json')).slice(0,20); } catch {}
          for (const file of pending) {
            const pass = await transport.tool('remember', { action:'backfill', params:JSON.parse(fs.readFileSync(path.join(backfillQueue,file),'utf8')) });
            if (!pass.change_budget_reached) try { fs.unlinkSync(path.join(backfillQueue,file)); } catch {}
          }
        }
      }
    }
    let context = ''; let packet = {};
    let cause = trigger(prompt, phrases);
    if (['session', 'prompt', 'tool'].includes(event)) {
      // The unified session action preserves exposure dedup and the per-turn budget.
      packet = await transport.tool('session', { action:'start', memory_hook: { prompt, limit: event === 'session' ? 8 : cause.limit, trigger: cause,
        session, host: client, event, source: typeof input.source === 'string' ? input.source : '' } });
      if (!Array.isArray(packet.memories)) throw new Error('binary lacks v6.2 session memory hooks');
      cause = trigger(prompt, phrases, packet.entities || []);
      context = formatContext(packet);
    }
    // A local receipt records only trigger labels and handles; never full prompt/transcript.
    const receipt = { timestamp: new Date().toISOString(), event, client, trigger: cause, saved,
      memories: (packet.memories || []).map(n => n.id), backfill_receipts: (packet.backfill_receipts || []).map(r => r.receipt_id), elapsed_ms: performance.now() - started };
    fs.appendFileSync(path.join(dir, 'receipts.jsonl'), JSON.stringify(receipt) + '\n', { mode: 0o600 });
    return output(client, event, context);
  } catch (error) { log(error.code || error.message); return {}; }
  finally { if (transport) transport.close(); }
}
async function main() {
  const args = process.argv.slice(2); const value = flag => args[args.indexOf(flag) + 1];
  const failOpen = setTimeout(() => { process.stdout.write('{}\n'); process.exit(0); }, 205);
  let raw = ''; for await (const chunk of process.stdin) { raw += chunk; if (raw.length > 65536) throw new Error('hook input too large'); }
  const result = await run({ event: value('--event'), client: value('--client'), configFile: value('--config'), input: JSON.parse(raw || '{}') });
  clearTimeout(failOpen); process.stdout.write(JSON.stringify(result) + '\n');
}
module.exports = { thingWords, thingKey, thingKeys, dayOfText, yearlyOfText, indexTags, run, rpc, explicitFact, extractFacts, sentences, formatContext, output, admissionOf, linksFor, drainQueue, ingestItem };
if (require.main === module) main().catch(() => { process.stdout.write('{}\n'); process.exit(0); });
