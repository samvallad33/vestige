# Reproducible Charm VHS terminal recording script for Vestige releases
# Run: vhs assets/vhs/vestige-demo.tape

Output assets/vhs/vestige-demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1280
Set Height 720
Set Theme "TokyoNight"
Set Padding 24
Set TypingSpeed 40ms

Type "vestige card --repo . --since 7d"
Enter
Sleep 2.5s

Type "vestige redteam"
Enter
Sleep 2.5s

Type "vestige streak --repo . --badge"
Enter
Sleep 2s

Type "vestige lease set --deny-cmd retrieval_bench --max-repeat-fail 2"
Enter
Sleep 2s

Type "vestige steer 'Wire Stage 12 before benchmarking' --halt"
Enter
Sleep 2.5s
