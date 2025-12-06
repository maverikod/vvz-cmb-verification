# Implementation Plan vs Theory: Phase 2–6 Differences

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com  
**Date:** 2024-12-03

**Note:** This file lists **explicit differences** (or their отсутствие) between the
implementation plan (Phase 2–6) and the theoretical/technical specification:

- Theory: `docs/ALL.md` (7D Θ-field, JW–CMB, JW–CMB2)
- Tech Spec: `docs/tech_spec-new.md` (Modules A–D, formulas 2.1–2.4, §3.3–3.4)
- Plan: `docs/implementation_plan/implementation_plan.md` + step READMEs

Search in theory used `docs/search_theory_index.py` with tags/phrases:
`JW–CMB`, `JW–CMB2`, `ΔT`, `Δω/ω`, `C_l`, `Module D`, etc.

---

## Legend

- **Status: OK** – план шаг в точности повторяет ТЗ/теорию или использует
  эквивалентную постановку.
- **Status: Method-only difference** – отличается лишь методом реализации, но
  **строго ссылается на ту же формулу** и не вводит запрещённой физики.
- **Status: Risk (impl)** – план корректен, но при реализации требуется особо
  следить за отсутствием классических формул (отмечено уже в FORMULA_ANALYSIS.md).

На момент этого отчёта **все реальные расхождения уровня “формула не та”
были найдены и исправлены в коде (см. `CODE_THEORY_COMPLIANCE_ISSUES.md`)**.
В планах Phase 2–6 явных теоретических противоречий **не осталось**.

---

## Phase 2: CMB Map Reconstruction

### Step 2.1 – CMB Reconstruction Core (Module A)

- **Plan:**
  - Использует формулу 2.1: \(\Delta T = (\Delta\omega/\omega_{\rm CMB}) T_0\).
  - Глубина узла: \(\Delta\omega(n^\hat{}) = \omega(n^\hat{}) - \omega_{\min}(n^\hat{})\).
  - Источник микроструктуры: спектр \(\rho_\Theta(\omega,t)\) (CMB2.1).
- **Theory / Tech Spec:**
  - tech_spec-new.md §2.1, §3.1 (Module A) и All.md JW–CMB.6, JW–CMB2.6.
- **Differences:**
  - **Status: OK** – план дословно повторяет формулы и требования Module A.
  - Запрещённые элементы (FRW, плазма, BAO, mass-terms, Silk) явно вынесены в
    `Forbidden Elements` → соответствует §4 и §6 tech_spec-new.md.

### Step 2.2 – CMB Map Validation

- **Plan:**
  - Сравнение с ACT DR6.02 по: arcmin-структуре (2–5′) и амплитуде 20–30 μK.
  - Использует только статистику (корреляции, разности), без физики ΛCDM.
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (конечные результаты 1, 2, 8).
  - All.md JW–CMB, главы 13.1–13.6 (узлы, масштаб, амплитуда).
- **Differences:**
  - **Status: OK** – чисто валидационный шаг, не вводит новых физических
    предположений.

### Step 2.3 – Node-to-Map Mapping

- **Plan:**
  - Проецирует узлы с z≈1100 в координаты \((\theta,\phi)\) сегодняшнего неба.
  - Явно требует: использовать **ТОЛЬКО** фазовые параметры \(\omega_{\min}(t),
    \omega_{\\text{macro}}(t)\); **запрещает** FRW, ΛCDM и “proper motion”.
- **Theory / Tech Spec:**
  - tech_spec-new.md §3.3 (Module C) и §4 (материя не влияет на ВБП).
  - All.md JW–CMB, §13.2–13.3 (узлы как источник CMB, проекция на сферу).
- **Differences:**
  - **Status: Risk (impl)** – сам текст шага согласован, но
    FORMULA_ANALYSIS.md (стр. 271–274) помечает: *при реализации*
    необходимо убедиться, что нет скрытых классических формул
    проекции (FRW и др.).
  - Теоретического расхождения в README **нет**, есть только
    акцент на аккуратной реализации.

---

## Phase 3: Power Spectrum Generation

### Step 3.1 – Power Spectrum Calculation (Module B)

- **Plan:**
  - Использует формулы 2.2–2.4:
    - \(\ell \approx \pi D\,\omega\) (2.2)
    - \(C_\ell \propto \rho_\Theta(\omega(\ell))/\ell^2\) (2.3)
    - \(C_\ell \propto \ell^{-2}\) (2.4).
  - **Ключевое:** считает \(C_\ell\) **напрямую из** \(\rho_\Theta(\omega,t)\),
    а не из карты через стандартную формулу
    \(C_\ell = \frac{1}{2\ell+1} \sum_m |a_{\ell m}|^2\).
  - В README есть заметка: это эквивалентно CMB2.3 и эффективнее, SH-разложение
    можно тестировать отдельно.
- **Theory / Tech Spec:**
  - tech_spec-new.md §2.2–2.4, §3.2 (Module B) – формулы совпадают.
  - All.md JW–CMB2.6–2.8 – вывод \(C_\ell\) из спектра Θ без Silk-damping.
- **Differences:**
  - **Status: Method-only difference.**
    - Module B в ТЗ упоминает “Spherical Harmonic Decomposition” карты.
    - План и README шага 3.1 выбирают **аналитически эквивалентный** путь:
      прямой расчёт \(C_\ell\) из \(\rho_\Theta(\omega,t)\) через 2.2–2.3.
    - FORMULA_ANALYSIS.md (§2.4, 3.1) уже помечает это как допустимую замену
      (эквивалентная теория, более эффективная реализация).

### Step 3.2 – High-l Sub-peaks Analysis

- **Plan:**
  - Извлекает биения из \(\omega_{\min}(t)\) и через \(\ell = \pi D \omega\)
    маппит их в sub-peaks \(C_\ell\), включая пик ℓ≈4500–6000.
  - Использует только Θ‑параметры; plasma / Silk не вводит.
- **Theory / Tech Spec:**
  - tech_spec-new.md §3.2 (Module B: sub-peaks).
  - All.md JW–CMB2.7–2.8 (high‑l хвост и sub‑peaks от биений \(\omega_{\min}\)).
- **Differences:**
  - **Status: OK** – 1:1 с текстом ТЗ и теории.

### Step 3.3 – Spectrum Comparison

- **Plan:**
  - Сравнивает рассчитанный \(C_\ell\) с ACT DR6.02 спектрами через χ² и
    другие метрики; явно запрещает Silk damping и классические модели.
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (конечные результаты по спектру).
- **Differences:**
  - **Status: OK / Risk (impl)** –
    - Физика берётся только из Θ‑модели; χ² выступает чисто статистическим
      инструментом (FORMULA_ANALYSIS.md, стр. 272–275).
    - В README нет введения запрещённой физики; риск только в том, как
      именно будет реализована статистика (следить при кодинге).

---

## Phase 4: CMB–LSS Correlation

### Step 4.1 – Correlation Analysis Core

- **Plan:**
  - CMB–LSS корреляции на 10–12 Mpc; использует Pearson‑корреляции и
    стандартную статистику; в `Forbidden Elements` запрещены V(φ), mass‑terms,
    FRW/ΛCDM.
- **Theory / Tech Spec:**
  - tech_spec-new.md §3.4 (Module D, вход CMB карта и LSS карта).
  - All.md JW–CMB, §13.7 (корреляция CMB→LSS).
- **Differences:**
  - **Status: OK / Risk (impl)** –
    - FORMULA_ANALYSIS.md помечает: важно, чтобы корреляции считались
      между CMB из Θ‑поля и LSS данными, без подмешивания классических
      космологических формул.
    - README это учитывает (запрет классики); теоретического конфликта нет.

### Step 4.2 – Phi-Split Analysis

- **Plan:**
  - φ‑split, enhancement = correlation_positive / correlation_negative,
    явно ссылается на JW–CMB2.8 (m‑modes).
- **Theory / Tech Spec:**
  - All.md JW–CMB2.8 (φ‑split и усиление сигнала из-за m‑мод).
- **Differences:**
  - **Status: OK** – полностью следует предсказанию теории.

### Step 4.3 – Node–LSS Mapping (Module D)

- **Plan:**
  - Реализует требования Module D: максимизация перекрытия с филаментами,
    U3/U1 по силе узла, жёсткий запрет “материя создаёт узлы” и любых T_{μν}.
- **Theory / Tech Spec:**
  - tech_spec-new.md §3.4 (Module D) и §4 (matter does NOT influence Θ‑field).
  - All.md §13.7 (цепочка CMB → LSS → галактики).
- **Differences:**
  - **Status: OK** – прямое соответствие тексту ТЗ и теории.

---

## Phase 5: ACT/SPT Predictions

### Step 5.1 – High‑l Peak Prediction

- **Plan:**
  - Предсказывает пик ℓ≈4500–6000 на основе Θ‑эволюции и sub‑peaks; никаких
    классических peak‑моделей.
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (пункты 3, 5).
  - All.md JW–CMB2.7–2.8.
- **Differences:**
  - **Status: OK** – план следует теории.

### Step 5.2 – Frequency Invariance Test

- **Plan:**
  - Тест ахроматичности (90–350 GHz) через кросс‑спектры, запрет
    frequency‑dependent “моделей CMB” (ахроматичность фундаментальна).
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (п. 3: “Подтверждение ахроматичности”).
  - All.md JW–CMB, §13.4–13.5 (ахроматичность как геометрическая).
- **Differences:**
  - **Status: OK** – чисто проверка предсказания, без добавления
    чужой физики.

### Step 5.3 – Predictions Report

- **Plan:**
  - Агрегирует результаты 5.1, 5.2, 3.2, 4.2; в запрещённых элементах
    исключает классические модели.
- **Theory / Tech Spec:**
  - Служебный отчётный модуль; физику не меняет.
- **Differences:**
  - **Status: OK**.

---

## Phase 6: Chain Verification

### Step 6.1 – Cluster Plateau Analysis

- **Plan:**
  - Анализ наклонов плато кластеров и их корреляции с направлениями узлов CMB.
  - Запрещает классические гравитационные модели (mass‑terms, potentials).
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (конечные результаты 7–8 по цепочке).
  - All.md §13.7 (связь узла с наклоном плато кластера).
- **Differences:**
  - **Status: OK** – соответствует теоретическому описанию цепочки.

### Step 6.2 – Galaxy Distribution Analysis

- **Plan:**
  - SWI, χ₆ и распределения U1/U2/U3 по направлениям узлов; запрет
    классических моделей формирования галактик.
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (п. 8) и Module D.
  - All.md §13.7 (тип галактик ↔ сила узла).
- **Differences:**
  - **Status: OK**.

### Step 6.3 – Chain Verification Report

- **Plan:**
  - Собирает всё: CMB → LSS → clusters → galaxies и валидирует цепочку.
  - Запрещает “classical chain models”, оставляя только Θ‑цепочку.
- **Theory / Tech Spec:**
  - tech_spec-new.md §5 (пункты 7–8), All.md §13.7.
- **Differences:**
  - **Status: OK** – отчётный модуль, физику не меняет.

---

## Summary

1. **Формальные формулы (2.1–2.4) и модули A–D** – полностью согласованы между
   теорией, tech_spec-new.md и планом (Phase 2–6).
2. **Единственное реальное расхождение** было в реализации формулы 2.1 в коде
   (`map_depth_to_temperature`), и оно уже исправлено (см.
   `CODE_THEORY_COMPLIANCE_ISSUES.md`). В ТЕКСТЕ плана (README/implementation_plan)
   формула изначально была корректной.
3. **Методологическое отличие** есть только у Step 3.1: план использует
   прямой расчёт \(C_\ell\) из \(\rho_\Theta(\omega,t)\) вместо явного
   SH-разложения карты, но это строго эквивалентно Formule 2.3 и отмечено как
   допустимое в FORMULA_ANALYSIS.md.
4. Остальные шаги Phase 2–6 либо служебные (валидация, отчёты), либо используют
   только Θ‑параметры и статистические инструменты, **без** введения запрещённой
   физики (плазма, FRW, ΛCDM, mass‑terms, potentials, Silk‑damping).

**Вывод:** с точностью до методического выбора в Step 3.1 (отмеченного как
эквивалентный и допустимый) план Phase 2–6 **полностью согласован** с теорией
7D Θ‑поля и tech_spec-new.md.
