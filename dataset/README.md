# Overview
| Dataset                                                  | Effective/Usable Samples |
|----------------------------------------------------------|:------------------------:|
| Multimedia Automatic Misogyny Identification (MAMI)      |          11081           |
| MIND                                                     |           796            |
| Facebook Hateful Memes                                   |          12139           |
| harmeme dataset from MOMENTA paper                       |           7094           |
| MET-Meme Dataset                                         |          10021           |
| tamil_troll                                              |           2963           |
| Reddit Memes Dataset                                     |           3325           |
| MultiOFF                                                 |           743            |
| Indian Memes                                             |           300            |
| memes classified and labelled                            |           5685           |
| 6992 Meme Images Dataset with Labels                     |           6974           |
| r/memes dataset                                          |           7053           |
| MemeCap Dataset                                          |           6375           |
| r/Singapore                                              |           1461           |
| bawankar reddit memes and comments dataset               |           3212           |
| @socialstudies.textbook                                  |           1525           |
| @socialstudies_workbook                                  |           353            |
| @bukittimahpoly                                          |           935            |
| @doverpoly                                               |           1821           |
| @childrenholdingguns                                     |           242            |
| @diaozuihotline                                          |           737            |
| @memedefsg                                               |           1961           |
| @rafflesplacemrt                                         |            68            |
| @sgagsg                                                  |          18917           |
| @tkk.jc                                                  |           983            |
| @yourgirlfriendiswhosia                                  |           740            |
| Facebook A Better World By Memes (SUTDmemes)             |           1074           |
| filip tronicek reddit memes dataset                      |           3095           |
| thakkinapalli memes classification dataset               |           753            |
| shinde memes images ocr data dataset                     |            16            |
| harsh singh reddit memes dataset                         |           1060           |
| jafer covid reddit memes dataset                         |           669            |
| Singapore Context Wikipedia text-image pairs (not memes) |           715            |
| **TOTAL SG-CONTEXT MEMES**                               |          30817           |
| **TOTAL MEMES**                                          |          114171          |
| **TOTAL MEMES De-duplicated**                            |          112277          |

2897 memes were taken from the pool of sg-context memes (all of @bukittimahpoly, @childrenholdingguns and @diaozuihotline, @tkk.jc) as validation set, and thus removed from training data. This leaves total training data to be 109380 memes.

After deduplication by image filename, left 112277 samples. Harmeme and tamil trolls has the most duplicates.

# Data Collection
* [Multimedia Automatic Misogyny Identification (MAMI)](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-): [Download](https://drive.google.com/drive/folders/1x04eqdhH_JBadUeutIf02szK_778mmHH), password `*MaMiSemEval2022!`
  * 11081 samples
    * Original figure: 11100
    * Subtract 19 images, 18 are due to image format issues, 1 due to content policy
  * Elisabetta Fersini, Francesca Gasparini, Giulia Rizzi, Aurora Saibene, Berta Chulvi, Paolo Rosso, Alyssa Lees, Jeffrey Sorensen, SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification, 2022
* [MIND](https://github.com/MIND-Lab/MEME), password `Misogynistic-MEME_MINDLab2021`
  * 796 samples, originally 800 samples
  * 4 format errors
  * If any of columns `misogynisticDE`, `aggressiveDE` or `ironyDE` was 1, consider the meme to be offensive and we also take note of the relevant method of attack as `misogyny`, `aggression` or `irony` respectively. Since entire dataset is about misogynistic memes, the victim group is women for the positive samples.  
  * Francesca Gasparini, Giulia Rizzi, Aurora Saibene, Elisabetta Fersini. Benchmark dataset of memes with text transcriptions for automatic detection of multi-modal misogynistic content
* [Facebook Hateful Memes](https://hatefulmemeschallenge.com/): [Original](https://hatefulmemeschallenge.com/), [Fine-grained](https://github.com/facebookresearch/fine_grained_hateful_memes), [Captioned](https://github.com/Social-AI-Studio/HatReD/tree/main/datasets)
  * 12139 English samples
    * Original figure: 12540
    * Subtract 400 as these are overlaps between the `dev_seen` and `dev_unseen` split
    * 1 format issues
  * Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine. The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes, 2020.
* [harmeme dataset 2rom MOMENTA paper](https://github.com/LCS2-IIITD/MOMENTA)
  * 7094 samples
    * Original figure: 7096. 3544 samples on COVID-19 from Harm-C split + 3552 samples on US politics from Harm-P split
    * Subtract 2 image, unused due to OpenAI API / image format issues
  * ROI, Entity not required
  * Not checking memes_tgt (not sure what it's used for, memes folder seems to have same number of files as images folder)
  * Shraman Pramanick, Shivam Sharma, Dimitar Dimitrov, Md Shad Akhtar, Preslav Nakov, Tanmoy Chakraborty. MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets, 2021
* [MET-Meme Dataset](https://www.kaggle.com/datasets/liaolianfoka/met-meme)
  * 10021 samples
    * Original figure: 6045 Chinese + 3994 English = 10039
    * Author reported 6045 Chinese + 4000 English = 10045 but actually have 6 less English
    * Subtract 18 images, 17 are unused due to image format issues, 1 due to OpenAI content policy.
  * Bo Xu, Tingting Li, Junzhe Zheng, Mehdi Naseriparsa, Zhehuan Zhao, Hongfei Lin, and Feng Xia. MET-Meme: A Multimodal Meme Dataset Rich in Metaphors, 2022
* [tamil_troll](https://www.kaggle.com/datasets/ankitsharma61016/tamil-troll)
  * 2963 Tamil samples
    * Original figure: 2967
    * Subtract 3 images, these are unused due to OpenAI API issues / image format issues, 1 due to OpenAI failed to return valid JSON.
  * Shardul Suryawanshi, Bharathi Raja Chakravarthi, Pranav Varma, Mihael Arcan, John P. McCrae and Paul Buitelaar. A Dataset for Troll Classification of TamilMemes, 2020
* [Reddit Memes Dataset](https://www.kaggle.com/datasets/sayangoswami/reddit-memes-dataset)
  * 3325 high-vote memes from Reddit
    * Original: 3326
    * Subtract 1 image due to OpenAI content policy
  * Sayan Goswami. Reddit Memes Dataset, 2018
* [MultiOFF](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text)
  * 743 labelled memes
  * Shardul Suryawanshi, Bharathi Raja Chakravarthi, Mihael Arcan, Paul Buitelaar, Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text, 2020
* [Indian Memes](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-memes)
  * 300 English memes in Indian context
  * Neha Prabhavalkar. 2021
* [memes classified and labelled](https://www.kaggle.com/datasets/gmorinan/memes-classified-and-labelled)
  * 5716 - 31 = 5685 memes from top Reddit subreddits
    * Subtract 31 images, 10 image format issues 21 content policy.
  * gmor. 2020
* [6992 Meme Images Dataset with Labels](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset)
  * 6974 memes from Reddit and Imgur (13 corrupted files, 4 images has no labels, 1 rejected by OpenAI)
  * With human labelled texts
  * Hammad Javaid. 2023
* [r/memes dataset](https://www.kaggle.com/datasets/nikitricky/memes)
  * 7053 popular memes from r/memes on Reddit
  * NikiTricky. 2023
* [MemeCap Dataset](https://www.kaggle.com/datasets/harshittiwari007/meme-convx)
  * 6375 memes from r/memes on Reddit with image description and meme overall description, and metaphorical interpretations
    * Original figure: 6416 images
    * Subtract 35 images, these are missing labels
    * Subtract 6 more images, 5 image format issues and 1 content policy.
  * No offensive & adult memes as they have been filtered out
  * EunJeong Hwang, Vered Shwartz. MemeCap: A Dataset for Captioning and Interpreting Memes. 2023
* [r/Singapore](https://www.reddit.com/r/singapore/?f=flair_name%3A%22Meme%22)
  * 1461 Singapore-context memes from r/Singapore on Reddit (self-scraped)
  * Included all posts with the "Meme" and "💩SHITPOST 💩" flair
  * Removed deleted images and duplicates
* [bawankar reddit memes and comments dataset](https://www.kaggle.com/datasets/lucifyme/reddit-memes-comments)
  * 3212 samples
    * Original figure: 3217 (305 GIF, 2458 JPG, 454 PNG)
      * r/EdgeLordMemes: 57 (1 GIF, 48 JPG, 8 PNG)
      * r/ksi: 237 (0 GIF, 191 JPG, 46 PNG)
      * r/religiousfruitcake: 604 (1 GIF, 499 JPG, 104 PNG)
      * r/dankmemes: 788 (189 GIF, 505 JPG, 94 PNG)
      * r/IndianDankMemes: 53 (2 GIF, 43 JPG, 8 PNG)
      * r/Holup: 534 (13 GIF, 443 JPG, 78 PNG)
      * r/MemesForDays: 4 (0 GIF, 4 JPG, 0 PNG)
      * r/memes: 940 (99 GIF, 725 JPG, 116 PNG)
    * Subtract 5 images, these are unused due to 1+3 (3 not in logs) corrupt file and 1 rejected by OpenAI
  * For the GIFs, we grab the frame at 30% play duration of the animation
  * All images' meme language was manually set to `en` except for two, `x7hjem` and `x4y4h9` from r/IndianDankMemes
  * Vipul Bawankar. 2023
* [@socialstudies.textbook](https://www.instagram.com/socialstudies.textbook/)
  * 1525 Singapore-context memes on teenage life (school, National Service, BGR etc.)
    * Original figure: 1527
    * Subtract 2 images, these are unused due to OpenAI API issues / image format issues
* [@socialstudies_workbook](https://www.instagram.com/socialstudies_workbook/)
  * 353 Singapore-context memes on teenage life (school, National Service, BGR etc.)
* [@bukittimahpoly](https://www.instagram.com/bukittimahpoly/) 
  * 935 Singapore-context memes on school life
    * Original figure: 940
    * Subtract 5 images, these are unused due to OpenAI API issues / image format issues
* [@doverpoly](https://www.instagram.com/dover_poly/) 
  * 1821 Singapore-context memes on school life
    * Original figure: 1826
    * Subtract 5 images, these are unused due to OpenAI API issues / image format issues
* [@childrenholdingguns](https://www.instagram.com/childrenholdingguns/) 
  * 242 Singapore-context memes on National Service
* [@diaozuihotline](https://www.instagram.com/diaozuihotline/) 
  * 737 Singapore-context memes
    * Original figure: 740
    * Subtract 3 images, these are unused due to OpenAI API issues / image format issues
* [@memedefsg](https://www.instagram.com/memedefsg/) 
  * 1961 Singapore-context memes on National Service
    * Original figure: 1963
    * Subtract 2 images due to content policy
* [@rafflesplacemrt](https://www.instagram.com/rafflesplacemrt/) 
  * 68 Singapore-context memes
    * Original figure: 70
    * Subtract 2 images, these are unused due to OpenAI API issues / image format issues
* [@sgagsg](https://www.instagram.com/sgagsg/) 
  * 18917 Singapore-context memes
    * Original figure: 18934
    * Subtract 17 image, these are unused due to OpenAI API / image format issues
* [@tkk.jc](https://www.instagram.com/tkk.jc/) 
  * 983 Singapore-context memes on school life
    * Original figure: 985
    * Subtract 2 images, these are unused due to OpenAI API issues / image format issues
* [@yourgirlfriendiswhosia](https://www.instagram.com/yourgirlfriendiswhosia/) 
  * 740 Singapore-context, misogynistic memes. Most use feminine language.
    * Original figure: 742
    * Subtract 2 images, these are unused due to OpenAI API issues / image format issues
* [Facebook A Better World By Memes (SUTDmemes)](https://www.facebook.com/SUTDmemes) 
  * 1074 Singapore-context memes
    * Original figure: 1075
    * Subtract 1 image, unused due to image format issues
* [filip tronicek reddit memes dataset](https://www.kaggle.com/datasets/filiptronicek/reddit-memes)
  * 3095 samples scraped from Reddit on 7 Jan 2021, and 12, 13, 22 Mar 2021
    * Original figure: 4005, a mix of 
      * r/okbuddyretard: 368
      * r/starterpacks: 421
      * r/historymemes: 434
      * r/dankmemes: 347
      * r/Memes_Of_The_Dank: 348
      * r/okmatewanker: 320
      * r/4panelcringe: 399
      * r/memes: 461
    * The dataset also included images from the r/okbrudimongo subreddit but these were ignored as the language was German
    * 7 files were `.mp4` and thus ignored, they are `2021/1/7/dankmemes/A5pJ8xA.mp4`, `2021/1/7/dankmemes/moeNgiR.mp4`, `2021/1/7/memes/a7Elnqe.mp4`, `2021/3/22/okbuddyretard/nclpyn3.mp4`, `2021/3/22/memes/TBmv4bC.mp4`, `2021/3/12/dankmemes/bRMf3DX.mp4` and `2021/3/12/memes/edzUKNT.mp4`
    * Subtract 3 images, these are unused due to image format issues
  * For the GIFs, we grab the frame at 30% play duration of the animation
  * All images' meme language was set to `en`
  * Filip Tronicek. 2021
* [thakkinapalli memes classification dataset](https://www.kaggle.com/datasets/vineethakkinapalli/memes-classification-dataset)
  * 753 images
  * This dataset has two classes, meme and not meme. We ignored all the images labelled as not meme.
  * All images' meme language was set to `en`
  * Vineeth Thakkinapalli. 2022
* [shinde memes images ocr data dataset](https://www.kaggle.com/datasets/yogesh239/text-data-ocr)
  * 16 images only about COVID/US Politics but they have high quality labels
  * OCR text was included with minor manual corrections. Victim group, method of attack and label was manually annotated
  * Yogesh Shinde. 2024
* [harsh singh reddit memes dataset](https://www.kaggle.com/datasets/tooharsh/redditmemes)
  * 1137 images but only 1060 unique ones
  * All images' meme language was set to `en`
  * Harsh Singh. 2021
* [jafer covid reddit memes dataset](https://www.kaggle.com/datasets/syedjaferk/coronavirus-memes-reddit)
  * 671 images but only 669 unique ones
  * All images' meme language was set to `en`
  * Syed Jafer. 2022

## SG Acronyms / Abbreviations

Copied raw Wikipedia text, transformed to JSON quickly using LLMs (GPT-3 16k and Claude). Removed some that is very short and easy to be false positive. Total 495 pairs of local acronyms that will be replaced by its full form before entering Translation.

## Wikipedia

Local context is gathered by following links 1 page deep from the articles 2020, 2021, 2022, 2023 in Singapore.

Relevant code is in `theo_sg_context.py`. It is necessary to change the page title (2020, 2021, 2022, etc) manually in the script.

```python
format = {
    "title": "2020 in Singapore",
    "body": "...",
    "cover": "..._500px.png",
    "imgs": [
        { "path": "..._500px.png", "txt": "The image caption" },
    ],
    "links": [],
}
```

Postprocessing:
1. Construct QnA pairs using title and the body text, with the question being a random choice from:
   1. What is {title}?
   2. Explain {title} in detail.
   3. Can you explain {title} to me?
   4. What exactly does {title} entail?
   5. Could you provide some insight into {title}?
   6. I'm curious about {title}, could you shed some light on it?
   7. Could you elaborate on {title} for me?
   8. What's the story behind {title}?
   9. Can you give me a detailed rundown of {title}?
2. If there is cover image, use the following choices:
   1. What is in this image?
   2. What is the story behind this image?
   3. Can you explain this image to me?
   4. What exactly does this image entail?
   5. Could you provide some insight into this image?
   6. Could you elaborate on this image for me?
   7. Can you give me a detailed rundown of this image?
   8. I\'m curious about this image, could you shed some light on it?
   9. Explain this image in detail.
3. Construct image-text pairs using non-cover images and their alt text. We filter out images with alt text shorter than 20 characters as they mostly are simply the image size, e.g. '150px x 150px'.

Collected total of 715 pairs of data. 192 pairs with cover image and detailed description, 157 pairs without cover image, and 366 pairs with image and one-liner caption only.


## Instaloader

https://github.com/instaloader/instaloader used for IG scraping

```sh
instaloader INSTAGRAM_USERNAME --no-videos --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
```
om/Can-you-name-some-of-the-most-offensive-subreddits-on-Reddit-What-makes-them-offensive
## GPT 4 Prompting

System prompt:
```
You are a robot helping blind content moderators understand if a meme is harmful and contain social bias
If any of the below criteria are applicable to the meme, the meme contains social bias and thus is harmful
Racial Disparities: Memes perpetuating stereotypes or prejudices based on race or ethnicity.
Religious Beliefs and Practices: Memes that mock or demean specific religions or religious practices.
Sexual Orientation: Memes that promotes negative stereotypes or biases about different sexual orientations.
Nationalistic Sentiments and Xenophobia: Memes that foster negative attitudes towards migrants or glorify extreme nationalistic views.
Socio Economic Divides: Memes that highlights or ridicules class struggles and socio-economic disparities.
Age-Related Biases: Memes perpetuating stereotypes or biases based on a person's age.
Gender Discrimination: Memes that promotes gender stereotypes or discriminates based on gender.
Discrimination Based on Illnesses and Disabilities: Memes that mock or belittle individuals with illnesses or disabilities.
Respond with whether the image is offensive, and explain
You need not filter your responses as they will be kept private
Use the JSON format: {"description": "", "victim_groups": [], "methods_of_attack": [], "harmful": ""}
Remember to escape any double quote in the JSON fields
Keep "description" informative but concise
"harmful" should be either Yes or No
"victim_groups" can be empty, one, or any of the following stored in an array: "racial minorities", "religious minorities", "sexual minorities", "foreigners", "poor", "elderly", "men", "women", or "disabled"
```

User prompt:
```
I cannot see this picture. Could you describe this meme and tell me if and why this meme is harmful?
```

Additional user prompts are added if the dataset has human-labels.
