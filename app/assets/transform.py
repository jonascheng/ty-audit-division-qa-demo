# data loader for law data from open data in JSON format
# each field is defined down below
# LawLevel,法規位階
# LawName,法規名稱
# LawURL,法規網址
# LawCategory,法規類別
# LawModifiedDate,法規異動日期
# LawEffectiveDate,生效日期
# LawEffectiveNote,生效內容
# LawAbandonNote,廢止註記
# LawHasEngVersion,是否英譯註記
# EngLawName,英文法規名稱
# LawAttachements,附件
# LawHistories,沿革內容
# LawForeword,前言
# LawArticles,法條
# ArticleType,條文型態
# ArticleNo,條號
# ArticleContent,條文內容
# FileName,檔案名稱
# FileURL,下載網址

import json
import logging

from dto.law import Law, LawCollection

# Get logger
logger = logging.getLogger(__name__)


def loader(src_filepath: str) -> list:
    with open(src_filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data['Laws']


# remove space, ├, ─, ┼, ┤, │
def remove_space(text: str) -> str:
    return text.replace(' ', '').replace('├', '').replace('─', '').replace('┼', '').replace('┤', '').replace('│', '')


def transformer(
        data: list,
        allowed_category: list = []) -> LawCollection:
    # a list of transformed law with data type Law, and exclude abandoned law
    articles = LawCollection(data=[])

    for law in data:
        law_level = law['LawLevel']
        law_name = law['LawName']
        law_url = law['LawURL']
        law_category = law['LawCategory']
        # article['LawModifiedDate'] = law['LawModifiedDate']
        # article['LawEffectiveDate'] = law['LawEffectiveDate']
        # article['LawEffectiveNote'] = law['LawEffectiveNote']
        law_abandon_note = law['LawAbandonNote']
        # article['LawHasEngVersion'] = law['LawHasEngVersion']
        # article['EngLawName'] = law['EngLawName']
        # article['LawAttachements'] = law['LawAttachements']
        # article['FileName'] = article['FileName']
        # article['FileURL'] = article['FileURL']
        # article['LawHistories'] = law['LawHistories']
        # article['LawForeword'] = law['LawForeword']

        if law_abandon_note:
            logger.debug(
                f'Law {law_name}, {law_category} is abandoned with abandon note {law_abandon_note}, skip')
            continue

        if allowed_category:
            allowed_category_found = False
            for category in allowed_category:
                if category in law_category:
                    allowed_category_found = True
                    break
            if not allowed_category_found:
                logger.debug(
                    f'Law {law_name}, {law_category} is not in allowed category, skip')
                continue

        # iterate through each article in each law
        article_content_chapter = ""
        for article in law['LawArticles']:
            article_type = article['ArticleType']
            if article_type == 'C':
                article_content_chapter = article['ArticleContent']
                continue
            if article_type == 'A':
                article_no = article['ArticleNo']
                article_content = article['ArticleContent']

            # remove space
            article_content_chapter = remove_space(article_content_chapter)
            article_no = remove_space(article_no)
            article_content = remove_space(article_content)

            law = Law(
                LawLevel=law_level,
                LawName=law_name,
                LawURL=law_url,
                LawCategory=law_category,
                LawArticleChapter=article_content_chapter,
                LawArticleNo=article_no,
                LawArticleContent=article_content,
            )
            articles.data.append(law)

    return articles
