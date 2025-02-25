from io import BytesIO
import json
import re
import requests
from PIL import Image
from openai import OpenAI
import wikipedia
import pyvips
client = OpenAI()

f = open("sg_context/context.json", "r", encoding="utf-8")
global_output = json.load(f)
f.close()

format = {
    "title": "2020 in Singapore",
    "body": "...",
    "cover": "..._500px.png",
    "imgs": [
        { "path": "..._500px.png", "txt": "The image caption" },
    ],
    "links": [],
}

header_styles = [
    "== References ==", "=== References ===", "==== References ====",
    "==References==", "===References===", "====References====", 
]

def gather_images_with_captions(text):
    out = []
    matches = re.findall(r"\[\[File:(.+?)[|\]].+\|(.+?)\]\]", text, re.MULTILINE)
    for match in matches:
        out.append([match[0], match[1]])
        # print(f"File: {match[0]}, Caption: {match[1]}")
    return out

def strip_non_text(text):
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL) # Remove references
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL) # Remove tables
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE) # Strip table thing
    text = re.sub(r'<!--[\s\S\n]*?-->', '', text, flags=re.MULTILINE) #  Strip comments
    text = re.sub(r'\{\{(.*?)\}\}', "", text, flags=re.MULTILINE | re.DOTALL)
    text = text.replace("\n\n\n", "")
    text = text.replace("}}", "")
    return text

def save2png(img_bytes, name, max_longest_side=500):
    try:
        img = pyvips.Image.new_from_buffer(img_bytes, "")
        img = img.resize(max_longest_side / max(img.width, img.height))
        img.write_to_file(name)
        print(f"Image converted successfully to {name}")
    except Exception as e:
        print(f"Error converting image: {e}")

def gather_links(text):
    links = re.findall(r'\[\[([^\|\[\]]+?)(?:\|.*?)?\]\]', text)
    return links

def swap_links_syntax(text):
    def replace_link(match):
        parts = match.group(1).split('|')
        if len(parts) == 1:
            return parts[0]
        else:
            return parts[1]
    
    return re.sub(r'\[\[(.*?)\]\]', replace_link, text)

def relevant(text):
    print(text)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are to determine if the following user messages, comprising a title and short description, are significant events, persons or places in the context of Singapore. If so, output ===YES=== and nothing else. Otherwise, output ===NO===. It is crucial that you only include events, persons and places, and not general topics such as 'Covid-19', 'English language', and similar."},
            {"role": "user", "content": text},
        ]
    )
    print(completion.choices[0].message.content)
    if "===YES===" in completion.choices[0].message.content:
        return True
    return False
    # return True

def get_page(url, depth):
    try:
        cutoff = -1

        if depth == 2: return
        if url in global_output:
            print("=== Skip repeated ===")
            return

        page_data_full = {
            "title": url,
            "body": "...",
            "cover": None,
            "imgs": [],
            "links": [],
        }

        # page content
        r = requests.get(f"https://en.wikipedia.org/wiki/{url}?action=raw")
        wiki_text = str(r.content, encoding="utf-8")

        images_with_captions = gather_images_with_captions(wiki_text)
        print("Images with captions:")
        print(images_with_captions)

        stripped_text = strip_non_text(wiki_text)

        for style in header_styles:
            if style in stripped_text:
                cutoff = stripped_text.index(style)
        
        links = gather_links(stripped_text[:cutoff])
        page_data_full["links"] = links

        print(links)

        # the actual clean text.
        pg = wikipedia.page(url, auto_suggest=False)
        swapped_text = pg.content

        # relevant?
        if not relevant(swapped_text[:300]):
            print("Excluded")
            return
        print("Included")
        
        # thumbnail
        try:
            r = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&titles={url}&format=json&pithumbsize=1024")
            thumb_req = json.loads(r.content)["query"]["pages"]
            thumb_url = thumb_req[list(thumb_req.keys())[0]]["thumbnail"]["source"]
            
            r = requests.get(thumb_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
            })
            save2png(r.content, f"sg_context/img/{url}_thumb_500px.png")
            page_data_full["cover"] = f"{url}_thumb_500px.png"
        except:
            pass

        # split text...
        for style in header_styles:
            if style in swapped_text:
                cutoff = swapped_text.index(style)

        page_data_full["body"] = swapped_text[:cutoff]

        # page img
        for img in images_with_captions:
            if img[0].endswith("svg"): continue
            r = requests.get(f"https://commons.wikimedia.org/wiki/Special:FilePath/{img[0]}", headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
            })
            save2png(r.content, f"sg_context/img/{img[0]}_500px.png")
            page_data_full["imgs"].append({ "path": f"{img[0]}_500px.png", "txt": img[1] })

        # add to output
        global_output[url] = page_data_full

        # branch out.
        for link in links:
            if "Category:" in link: continue
            get_page(link, depth + 1)
    except Exception as e:
        print(e)

get_page("2023_in_Singapore", 0)

f = open("sg_context/context.json", "w", encoding="utf-8")
json.dump(global_output, f)
f.close()