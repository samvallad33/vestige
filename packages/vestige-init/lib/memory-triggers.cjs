'use strict';
// This is copied as one editable JSON file on install; phrases are literal, never regexes.
const defaults = {
  questions: ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose'],
  memory: ['remember', 'last time', 'you said', 'I told you', 'did I mention', 'again', 'still', 'as usual', 'like before'],
  time: ['yesterday', 'last week', 'a while ago', 'back when', 'earlier', 'before'],
  planning: ['should we', 'recommend', 'plan', 'pick', 'choose', 'what about', 'ideas for'],
  preferences: ['my favorite', 'I like', 'I hate', 'allergic', 'I prefer'],
  references: ['her', 'him', 'them', 'that place', 'that one', 'the usual'],
  engineering: ['regression', 'root cause', 'invariant', 'architecture', 'supersede', 'deprecated', 'failing test', 'deadlock', 'migration', 'schema', 'latency', 'postmortem'],
};
function contains(text, phrase) {
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?:^|[^\\p{L}\\p{N}_])${escaped}(?=$|[^\\p{L}\\p{N}_])`, 'iu').test(text);
}
function trigger(prompt, config = defaults, entities = []) {
  const matches = [];
  for (const [category, phrases] of Object.entries(config)) {
    if (!Array.isArray(phrases)) continue;
    for (const phrase of phrases.slice(0, 128)) {
      if (typeof phrase === 'string' && phrase.length > 0 && phrase.length <= 120 && contains(prompt, phrase)) matches.push({ category, phrase });
    }
  }
  for (const entity of entities) matches.push({ category: 'entity', phrase: entity });
  return { mode: matches.length ? 'triggered' : 'light', matches, limit: matches.length ? 8 : 3 };
}
module.exports = { defaults, contains, trigger };
