# Investigation Guide: Diary Analysis Tool

## The Discovery
You've discovered a series of diary entries from Marianne. These entries, dated August 2025, contain hidden patterns and recurring themes that may be crucial to understanding their design philosophy—and perhaps something more.

This tool analyzes the diary text to find word pairs (bigrams) that appear together with unusual frequency. These patterns aren't random; they're breadcrumbs left behind, whether intentionally or subconsciously. Phrases like "emotional intelligence" and "fail gracefully" might appear more often than chance would suggest. What do these patterns reveal?

---

## Quick Investigation Mode

### Begin Your Analysis:
1. Download the investigation files (see Step 1 below)
2. Double-click `simple_run.py` (or run `python simple_run.py` in terminal)
3. Watch as the tool uncovers patterns in the diary entries
4. Note the recurring phrases—they're not coincidental

The analysis will reveal:
- Hidden patterns in Marianne's entries
- Word pairs that appear with suspicious frequency
- Statistical anomalies that might be clues to something deeper
- CSV files containing your findings for further investigation

---

## Skilled Investigator's Protocol

### Step 1: Acquire the Investigation Files
1. Access the repository - click the green "Code" button on Github
2. Download the ZIP archive
3. Extract the files
4. You now possess the `nltk-bigram-analyzer` investigation toolkit

### Step 2: Verify Your Analysis Tools
1. Open your Terminal (Mac/Linux) or Command Prompt (Windows)
2. Check Python installation:
   ```
   python3 --version
   ```
3. Confirm you see `Python 3.x.x` - you'll need this for the investigation
4. If missing, acquire Python from [python.org](https://python.org) (version 3.8+)

### Step 3: Access the Investigation Directory
In your Terminal/Command Prompt:
1. Navigate to Download (or wherever you're storing the folder): `cd Downloads`
2. Enter the investigation folder: `cd nltk-bigram-analyzer`
3. Press Enter
4. You're now in position to begin the analysis

### Step 4: Initialize Investigation Tools

#### On Mac/Linux:
```bash
./install.sh
```
Confirm virtual environment setup when prompted (`y`).

#### On Windows:
```bash
pip install -r requirements.txt

python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])"
```

The tools are preparing to reveal what's hidden in the text.

### Step 5: Begin the Investigation
Execute your first pattern analysis:
```bash
python simple_run.py
```

You should see output like:
```
============================================================
 DIARY TEXT PATTERN ANALYZER 
============================================================

📖 Loading diary text...
✅ Loaded 8534 characters of diary entries

🔍 FULL PHRASE ANALYSIS (including 'in', 'the', etc.)
----------------------------------------
Total words: 1458

🎯 'unseen forces' appears 6 times!

📊 TOP 20 MOST COMMON PHRASES:
----------------------------------------
 1. unseen forces              6x ████████████ ⭐
 2. try harder                 4x ████████
 3. complicated position       3x ██████
   ...
```

**Pattern analysis complete. What does the repetition tell you?**

---

## What Did We Just Uncover?

The investigation tool:
1. **Accessed** the diary entries from August 2025
2. **Extracted** individual words, filtering out the noise
3. **Identified** consecutive word pairs that might hold meaning
4. **Calculated** frequency patterns—some too common to be coincidence
5. **Revealed** the most suspicious recurring phrases
6. **Documented** all findings in a CSV file for your investigation

The author writes about "noticing" things others miss. Is this self-awareness, or something else?

---

## Investigating Other Documents

### Method 1: Replace the Current Evidence
1. Navigate to the `data` folder
2. Open `diary_text.txt` in any text editor
3. Replace with new diary evidence when you discover it
4. Save the file
5. Run the analysis: `python simple_run.py`

### Method 2: Advanced Analysis Scripts
For deeper investigation, use the specialized scripts in `example-scripts/`:
   ```bash
   python example-scripts/analyze_diary.py
   ```
These provide more detailed thematic analysis and context examination.

Remember: Every text tells a story through its patterns. What story are you trying to uncover?

---

## Decoding the Evidence

### Pattern Recognition Output:
- **Total tokens**: Significant words extracted from the diary (both with and without common words)
- **Unique bigrams**: Distinct word pair combinations discovered
- **Top bigrams**: Most frequent patterns - these are rarely accidental
- **Statistical collocations**: Pairs appearing together beyond chance - potential clues
- **Special marker ⭐**: Highlights when the top bigram appears

### Evidence Archive:
Two CSV files document your findings:
1. `diary_analysis_with_all_words.csv` - includes common words like "unseen forces"
2. `diary_analysis_content_words.csv` - focuses on meaningful content words

Each file contains:
- Column A: The word pair (e.g., "try harder")
- Column B: First word
- Column C: Second word  
- Column D: Frequency count - how often does this pattern repeat?
- Column E & F: Statistical significance scores - higher values suggest intentional patterns

---

## Investigation FAQ

### Q: What's a bigram?
**A:** Two consecutive words forming a pattern. In the diary, "fail gracefully" is a bigram that appears suspiciously often.

### Q: Why are some words missing?
**A:** Common words ("the", "is", "and") are filtered to reveal meaningful patterns. But sometimes these "stopwords" hide clues...

### Q: I want more detailed analysis beyond the basic patterns
**A:** The main script `simple_run.py` already analyzes both WITH and WITHOUT common words, specifically tracking the top bigram. For even deeper analysis, use:
```bash
python example-scripts/analyze_diary_complete.py
```
This provides extensive thematic breakdowns and statistical correlations.

### Q: Can I analyze multiple files?
**A:** Yes, but you'll need to run the command once for each file.

### Q: What's PMI, Chi-square, etc?
**A:** Statistical methods that detect non-random patterns. When words appear together more than chance predicts, it suggests intentional placement.

---

## Troubleshooting the Investigation

### "Command not found" error
- Verify your location in the directory structure (`pwd` or `cd`)
- Windows investigators: use `python` instead of `python3`

### "No module named nltk" error
- Investigation tools incomplete. Reinitialize:
  ```bash
  pip install -r requirements.txt
  ```

### "LookupError" from NLTK
- Missing language analysis data. Download:
  ```bash
  python -c "import nltk; nltk.download('all')"
  ```

Every error might be intentional. The diary author mentions "failing gracefully" - perhaps errors are part of the design?

---

## Deeper Investigation

Once you've uncovered the initial patterns:

1. **Cross-reference findings**: Compare diary entries from different dates—do patterns evolve?
2. **Look for anomalies**: Which entries break the pattern? Those might be most revealing
3. **Track theme evolution**: How do concepts like "noticing" and "failing" connect?
4. **Question everything**: Why does the author mention her top bigrams so often?

---

## Need More Clues?

1. **Examine README.md** for technical investigation methods
2. **Study the `example-scripts` folder** - particularly `analyze_diary.py` for deeper thematic analysis
3. **Investigate source code** in `src` - the comments might contain hints
4. **Report findings on GitHub** if you discover something significant

Remember: Nothing is a coincidence. Every pattern, every repetition, every "failure" might be a designed experience. The author writes about "designing for people who notice." 

You noticed. Now what?
