# Raw data

One JSON array per court, as scraped by `scripts/extract_jurisprudencia.py`. Each record has fields including `ImportJurisprudenciaId`, `Sumario`, `Texto`, `TipoDecisao`, `Observacoes`, `NumeroProcesso`, `Data`.

## Court codes

| Code | Court |
|---|---|
| STJ | Supremo Tribunal de Justiça |
| STA | Supremo Tribunal Administrativo |
| TRC | Tribunal da Relação de Coimbra |
| TRE | Tribunal da Relação de Évora |
| TRG | Tribunal da Relação de Guimarães |
| TRL | Tribunal da Relação de Lisboa |
| TRP | Tribunal da Relação do Porto |
| TCAN | Tribunal Central Administrativo Norte |
| TCAS | Tribunal Central Administrativo Sul |
| TConflitos | Tribunal de Conflitos |

## Groupings

- **Group 1**: STJ, STA
- **Group 2**: TCAN, TCAS
- **Group 3 (`grupo3`)**: TRC, TRE, TRG, TRL, TRP — the group actually used for modeling (see `data/2_processed/preprocessed_grupo3.csv`)
