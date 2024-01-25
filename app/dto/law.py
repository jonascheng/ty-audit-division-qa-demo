from dataclasses import dataclass
from dataclass_wizard import JSONListWizard, JSONFileWizard, JSONSerializable, json_field


@dataclass(frozen=True)
class Law(JSONFileWizard):
    LawLevel: str = json_field('LawLevel', all=True)
    LawName: str = json_field('LawName', all=True)
    LawURL: str = json_field('LawURL', all=True)
    LawCategory: str = json_field('LawCategory', all=True)
    LawArticleChapter: str = json_field('LawArticleChapter', all=True)
    LawArticleNo: str = json_field('LawArticleNo', all=True)
    LawArticleContent: str = json_field('LawArticleContent', all=True)


@dataclass(frozen=True)
class LawCollection(JSONListWizard, JSONFileWizard, JSONSerializable):
    data: list[Law]
