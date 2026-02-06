# Analisi Progetto: zero_cost_hunter

## Sintesi
Il progetto e, di fatto, una piattaforma di trading e portfolio intelligence AI-assisted a basso costo operativo, non un semplice bot.

### Obiettivo reale
1. Scansionare mercato crypto/azionario con fonti gratuite.
2. Generare segnali e decisioni con un layer AI multi-agente.
3. Eseguire ribilanciamento e controllo rischio.
4. Esporre tutto via Telegram/Webhook + dashboard.

### Evidenze principali (file chiave)
- Visione e feature dichiarate: `README.md:3`, `README.md:15`
- Orchestrazione pipeline: `main.py:122`
- Superficie API/Bot: `api/webhook.py:559`
- Job schedulati su GitHub Actions: `.github/workflows/market_scan.yml:4`
- Routing deploy su Vercel: `vercel.json:4`

## Funzionamento End-to-End
1. Arrivo comandi da Telegram via webhook.
2. Dispatch dei task pesanti a GitHub Actions (`hunt`, `rebalance`, `analyze`, `trainml`, `backtest`).
3. Raccolta dati mercato/news/social/on-chain.
4. Decisioni tramite motore AI multi-modulo (`brain`, `council`, `critic`, `consensus_engine`, `constraint_engine`).
5. Layer predittivo ML custom (`ml_predictor`).
6. Persistenza SQLite con migrazioni versionate.
7. Output su dashboard e notifiche.

## Stato di maturita
- Test eseguiti: `127 passed`
- Coverage totale: `43%`
- Copertura buona su alcuni moduli core (es. `consensus_engine`, `sentinel`, `position_watchdog`)
- Copertura bassa su moduli grandi/critici:
  - `api/webhook.py` ~11%
  - `db_handler.py` ~18%
  - `brain.py` ~26%

Interpretazione: base ingegneristica valida, ma rischio operativo nei punti ad alta complessita e bassa copertura.

## Problemi concreti riscontrati
1. Bug probabile su metrica news: variabile non definita in `main.py:1351` (`unique_news_items_list`), con possibile errore silenziato da `try/except`.
2. Incoerenza API sentiment macro in `economist.py:242` e `economist.py:327`: chiamata a `Insider.get_fear_greed()` non allineata ai metodi disponibili in `insider.py`.
3. Metodo duplicato in `db_handler.py:176` e `db_handler.py:374` (`update_asset_quantity`), rischio shadowing.
4. Embedding memory di fatto disattivati in `memory.py:34` e `memory.py:43`.

## Evoluzioni consigliate (roadmap pragmatica)
1. **Hardening dati + dry-run totale (priorita massima, 1-2 settimane)**
   - Rendere `analyze`, `rebalance`, `trainml` eseguibili in modalita test senza side effect reali (Telegram, write DB, chiamate AI a consumo).
   - Hardening ticker/data quality (gestione ticker delisted/invalidi, fallback robusti, errori non silenziati).
   - KPI: zero side effect in dry-run, zero failure bloccanti su ticker sporchi.

2. **Rebalance con ottimizzazione vincolata (2-4 settimane)**
   - Evolvere da suggerimenti euristici a ottimizzazione con vincoli: costo/fee, impatto fiscale, concentrazione, correlazione, anti-churn.
   - Output in piano ordini eseguibile (sequenza e sizing buy/sell).
   - KPI: meno turnover inutile, migliore efficienza netta post-costi.

3. **Maturita TrainML (walk-forward + model gating) (2-4 settimane)**
   - Validation temporale realistica (walk-forward) e promozione modello solo se supera baseline.
   - Registry champion/challenger + rollback automatico su degrado.
   - KPI: metriche out-of-sample stabili, niente promozioni peggiorative.

4. **Osservabilita e KPI decisionali (2-3 settimane)**
   - Tracing strutturato end-to-end e dashboard quality: hit-rate, drawdown, precisione segnali, stabilita modello.
   - Report run standardizzati per `hunt`, `analyze`, `rebalance`, `trainml`.
   - KPI: diagnosi incidenti piu rapida e decisioni misurabili.

5. **Refactor architetturale + evoluzione prodotto (continuo)**
   - Modularizzazione progressiva file monolitici (`webhook.py`, `main.py`, `brain.py`, `db_handler.py`).
   - Feature evolutive: multi-account, policy rischio per profilo, explainability, simulation-first mode.

## Priorita consigliata
1. Hardening operativo (dati + dry-run completo)
2. Rebalance vincolato a costo/tasse/rischio
3. TrainML robusto con gating di promozione
4. Osservabilita/KPI run e qualita decisionale
5. Refactor ed evoluzione feature

---

Documento creato per consultazione continua e aggiornabile con nuove evidenze (test, benchmark, refactor, incidenti).
