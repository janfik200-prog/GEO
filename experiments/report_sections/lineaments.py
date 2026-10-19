# -*- coding: utf-8 -*-
"""Раздел «Линеаменты»: заглушка. Материал пользователь добавит позже.

Один слайд с пустой рамкой под будущее содержание; разделитель раздела ставит
главный скрипт.
"""
from src.slides_inst import caps, rect, textbox, run, GREY, GREY_L, PALE, BORDER_G, MARGIN, CONTENT_W

COUNT = 1


def build(deck):
    sl = deck.slide("ЛИНЕАМЕНТЫ", "ОТДЕЛЬНЫЙ ПРОЕКТ", "Линеаменты: материал в подготовке",
                    "Слайды раздела будут добавлены позже")
    rect(sl, MARGIN, 1.95, CONTENT_W, 2.45, PALE, BORDER_G, lw=1.0)
    caps(sl, MARGIN, 2.90, CONTENT_W, "МЕСТО ПОД СЛАЙДЫ РАЗДЕЛА", GREY_L, 10,
         align=__import__("pptx.enum.text", fromlist=["PP_ALIGN"]).PP_ALIGN.CENTER)
