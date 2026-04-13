# AgenticCodingBench demo GIF
# Usage: vhs demo.tape
#
# Adjust Sleep 45s if the benchmark takes longer/shorter
# (watch a real run first, then set Sleep to actual_time + 5s)

Output assets/demo.gif
Set Shell zsh
Set FontSize 15
Set Width 1200
Set Height 620
Set Padding 20
Set Theme { "name": "acb", "black": "#1e1e2e", "red": "#f38ba8", "green": "#a6e3a1", "yellow": "#f9e2af", "blue": "#89b4fa", "magenta": "#cba6f7", "cyan": "#89dceb", "white": "#cdd6f4", "brightBlack": "#585b70", "brightRed": "#f38ba8", "brightGreen": "#a6e3a1", "brightYellow": "#f9e2af", "brightBlue": "#89b4fa", "brightMagenta": "#cba6f7", "brightCyan": "#89dceb", "brightWhite": "#a6adc8", "background": "#1e1e2e", "foreground": "#cdd6f4", "selection": "#45475a", "cursor": "#f5e0dc" }
Set TypingSpeed 35ms
Set CursorBlink false

# Set env vars, clean prompt, and clear screen (all hidden from GIF)
Hide
Type "export SWARM_ONE_API_KEY=bdeaZWM6Y5 && export PS1='$ ' && clear"
Enter
Sleep 1s
Show

# Run the benchmark
Type "acb speed -e https://api.swarmone.ai/BO7p7sAE/v1/chat/completions -m qwen3 -k $SWARM_ONE_API_KEY --suite quick"
Sleep 800ms
Enter

# Wait for benchmark to complete — adjust this to match your actual runtime + 5s buffer
Sleep 30s

# Pause on the final table + verdict so viewers can read it
Sleep 6s
