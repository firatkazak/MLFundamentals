import json

with open("C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/spaCy/Data/Inputs/alice_unclean.txt", "r") as f:
    text = f.read()
print(text[:1000])

chapters = text.split("CHAPTER")[1:]
print(chapters[0][0:1000])

complete_text = []
for chapter in chapters:
    chapter = "CHAPTER" + chapter
    chapter = chapter.split("*       *       *       *       *       *       *")[0]
    paras = chapter.split("\n\n")
    chapter_num = paras[0]
    print(chapter_num)
    chapter_title = paras[1].strip()
    print(chapter_title)
    chapter_text = paras[2:]
    final_paragraphs = []
    for para in chapter_text:
        para = para.strip().replace("\n-", "").replace("\n", " ")
        while "  " in para:
            para = para.replace("  ", " ")
        if len(para) > 1:
            final_paragraphs.append(para)
    complete_text.append((chapter_num, chapter_title, final_paragraphs))

with open("C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/spaCy/Data/Inputs/alice.json", "w") as f:
    json.dump(complete_text, f, indent=4)
