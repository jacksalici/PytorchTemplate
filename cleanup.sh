#!/bin/bash
echo -e "                                        ___    ,'\"\"\"\"'.\n                                    ,\"\"\"   \"\"\"\"'      \`.\n                                   ,'        _.         \`._\n                                  ,'       ,'              \`\"\"\"'.\n                                 ,'    .-\"\"\`.    ,-'            \`.\n                                ,'    (        ,'                :\n                              ,'     ,'           __,            \`.\n                        ,\"\"\"\"'     .' ;-.    ,  ,'  \\             \`\"\"\".\n                      ,'           \`-(   \`._(_,'     )_                \`.\n                     ,'         ,---. \\ @ ;   \\ @ _,'                   \`.\n                ,-\"\"'         ,'      ,--'-    \`;'                       \`.\n               ,'            ,'      (      \`. ,'                          \`.\n               ;            ,'        \\    _,','                            \`.\n              ,'            ;          \`--'  ,'                              \`.\n             ,'             ;          __    (                    ,           \`.\n             ;              \`____...  \`78b   \`.                  ,'           ,'\n             ;    ...----'''' )  _.-  .d8P    \`.                ,'    ,'    ,'\n_....----''' '.        _..--\"_.-:.-' .'        \`.             ,''.   ,' \`--'\n              \`\" mGk \"\" _.-'' .-'\"-.:..___...--' \`-._      ,-\"\"'   \`-'\n        _.--'       _.-'    .'   .' .'               \"\"\"\"\"\"\n  __.-''        _.-'     .-'   .'  /\n '          _.-' .-'  .-'        .'\n        _.-'  .-'  .-' .'  .'   /\n    _.-'      .-'   .-'  .'   .'\n_.-'       .-'    .'   .'    /\n       _.-'    .-'   .'    .'\n    .-'            .'"
echo "  _____                    _       _        ____ _                             "
echo " |_   _|__ _ __ ___  _ __ | | __ _| |_ ___ / ___| | ___  __ _ _ __   ___ _ __  "
echo "   | |/ _ \ '_ \` _ \\| '_ \| |/ _\` | __/ _ \ |   | |/ _ \/ _\` | '_ \ / _ \ '__| "
echo "   | |  __/ | | | | | |_) | | (_| | ||  __/ |___| |  __/ (_| | | | |  __/ |    "
echo "   |_|\___|_| |_| |_| .__/|_|\__,_|\__\___|\____|_|\___|\__,_|_| |_|\___|_|    "
echo "                    |_|                                                        "
echo "READ CAREFULLY: This script will remove all the files regarding the toy sorting problem, delete the relative lines from the entry point and finally disappear itself."
echo "Press any key to continue or Ctrl+C to cancel."
read -n 1 -s

files=("configs/toySorter.yaml" "experiments/exp_toysort.py" "dataloaders/toy_dataloader.py" "models/transformer.py")

echo "Removing files..."
for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "Removed: $f"
    else
        echo "File not found: $f"
    fi
done

# --- Remove lines from run.py ---
TARGET_FILE="run.py"

lines_to_delete=(
    'from experiments.exp_toysort import ToySortExperiment, CustomCrossEntropyLoss'
    'from models.transformer import Transformer'
    '    "toysort": {"experiment": ToySortExperiment, "criterion": CustomCrossEntropyLoss},'
    '    "transformer": {"model": Transformer},'
    '        elif config.dataloader == "toysort_ds":'
    '            from dataloaders.toy_dataloader import get_dataloaders'
)

for line in "${lines_to_delete[@]}"; do
    if grep -qF "$line" "$TARGET_FILE"; then
        sed -i '' "/$line/d" "$TARGET_FILE"
        echo "Deleted line from $TARGET_FILE: $line"
    else
        echo "Line not found in $TARGET_FILE: $line"
    fi
done

for i in {5..1}; do
    echo -ne "This script will self-destruct in $i seconds.\r"
    sleep 1
done
echo ""

echo "Self-destructing now!"
rm -- "$0"
clear
echo "Cleanup done. Please add working dataloader, model, and experiment as needed. Enjoy!"

