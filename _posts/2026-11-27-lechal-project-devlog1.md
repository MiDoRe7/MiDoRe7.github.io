---
title: "[LecHAL 개발기 #1] 강의자료 PDF 파싱부터 하이브리드 인덱싱까지 (M1 구현 및 트러블슈팅)"
date: 2026-11-27 19:45:00 +0900
categories: [Project, LecHAL]
tags: [llm]
math: true
---

## 1. 프로젝트 소개: LecHAL

**LecHAL (Lecture Hallucination Verifier)** 은 대학 강의자료(PDF 슬라이드)를 기반으로 LLM 답변의 할루시네이션(환각) 여부를 판별하고, 답변의 각 주장(Claim)을 뒷받침하는 근거 슬라이드와 위치(Citation)를 제시하는 검증 도구이다.

전체 시스템은 다음과 같은 4단계 파이프라인으로 설계했다.

1. **Ingestion & Parsing (M1 - 완료)**: 과목별 PDF 슬라이드에서 텍스트·블록 좌표(BBox)·페이지 이미지를 추출하고, 수식·도표가 복잡한 페이지를 선별해 Markdown(LaTeX) 변환을 지원
2. **Hybrid Retrieval (M1 인덱싱 완료 / M2 검색 결합 예정)**: 한국어+수식 특화 BM25와 다국어 Dense 임베딩(BGE-M3)을 결합하고 Cross-Encoder Reranker로 근거 후보 정제
3. **Verification & Uncertainty (M2~M3 예정)**: LLM 답변을 원자적 주장(Claim)으로 분해해 NLI로 `Entailment / Neutral / Contradiction`을 판별하고, 불확실성 정량화(Entropy) 및 보정(Calibration) 적용
4. **Presentation (M4 예정)**: CLI 평가 도구 및 근거 영역을 하이라이트해 주는 Streamlit Split View 웹 UI

이번 글에서는 **M1(PDF 슬라이드 파싱 및 메타데이터 인덱싱)** 구현 구조와 실제 개발·리뷰 과정에서 겪은 트러블슈팅을 정리한다.

---

## 2. 기술 스택 및 핵심 설계 결정

| 항목 | 선택 | 선정 이유 |
|---|---|---|
| 환경 관리 | `uv` + Python 3.12 | 시스템 Python(3.14)과 `torch`, `chromadb` 간 휠 호환성 문제를 피하고 의존성을 재현 가능하게 관리 (`dense` 의존성은 optional extra로 분리) |
| PDF 파싱 | `PyMuPDF` (`pymupdf`) | 빠른 렌더링 속도, 텍스트 블록·폰트·좌표(BBox)·벡터 도형·이미지 메타데이터를 한 번에 추출 |
| 저장 포맷 | JSONL 원본 + 파생 인덱스 | 파싱/LaTeX 변환 결과를 `pages.jsonl`, `chunks.jsonl`에 원본 저장해 두고, 토크나이저나 임베딩 모델을 바꿔도 인덱스만 재생성 가능 |
| 키워드 검색 | `kiwipiepy` + `rank-bm25` | 한국어 조사 제거(형태소 분석)와 동시에 `\sum`, `\frac` 같은 LaTeX 명령 및 `∑, α, ∂` 수학 기호를 별도 토큰으로 보존 |
| 의미 검색 | `BAAI/bge-m3` + `ChromaDB` | 한국어·영어·수식이 혼재된 긴 슬라이드 텍스트(최대 8192 토큰) 임베딩 지원 |
| LaTeX 변환 | `LatexConverter` Protocol (`Claude` / `Noop`) | 백엔드를 교체 가능하도록 추상화하고, 복잡 페이지만 선별 변환 + 디스크 캐시 적용 (`--no-latex`로 오프라인/무과금 실행 가능) |

### 검색 단위와 좌표·ID 규약

- **검색 단위**: 슬라이드 1페이지 = 1 `Chunk`로 구성하되, 페이지 내 개별 블록 좌표(`block_bboxes`)를 함께 보존해 추후 UI 하이라이트에 활용할 수 있게 했다.
- **좌표계**: PDF 포인트(`1/72 inch`), 좌상단 원점(PyMuPDF 기본값)을 그대로 사용해 좌표 변환 버그를 없앴다. 이미지 픽셀 좌표는 `bbox × (dpi / 72)`로 바로 환산된다.
- **ID 규약**: 다과목 확장을 고려해 `page_id = "{course_id}:{lecture_id}:p{page_num}"`, `chunk_id = "{page_id}:c{idx}"` (1-based 페이지 번호)로 통일했다.

---

## 3. 현재까지 구현된 프로그램 구조

### 디렉터리 구조

```text
LecHAL/
├── configs/
│   └── default.yaml                 # 경로, 렌더링 DPI, 복잡도 임계값, LaTeX, 임베딩 설정
├── docs/
│   ├── ARCHITECTURE.md              # 전체 아키텍처 및 마일스톤
│   ├── DECISIONS.md                 # 주요 기술 의사결정 기록 (D-001 ~ D-006)
│   └── DEVLOG.md                    # 개발 일지 및 Next Steps
├── src/lechal/
│   ├── config.py                    # YAML → Pydantic Settings 로딩
│   ├── schemas.py                   # BBox, Block, PageRecord, Chunk 스키마 및 ID 헬퍼
│   ├── cli.py                       # Typer CLI (lechal ingest | index | inspect)
│   ├── ingestion/
│   │   ├── pdf_parser.py            # PyMuPDF 파싱 (텍스트/이미지 블록, bbox, PNG 렌더링)
│   │   ├── complexity.py            # 복잡 페이지 판별 휴리스틱
│   │   ├── latex_converter.py       # LatexConverter 프로토콜, Claude 비전 백엔드(+디스크 캐시)
│   │   ├── chunker.py               # PageRecord → 검색 단위 Chunk 변환
│   │   └── pipeline.py              # 과목 단위 PDF 일괄 처리 파이프라인
│   └── indexing/
│       ├── store.py                 # JSONL 원자적 입출력
│       ├── tokenizer.py             # Kiwi 형태소 + LaTeX 명령 + 수학 유니코드 토크나이저
│       ├── bm25_index.py            # BM25Okapi 인덱스 (JSON 직렬화)
│       └── dense_index.py           # BGE-M3 + ChromaDB 코사인 인덱스
└── tests/                           # 단위·통합·CLI 테스트 (36개)
```

### 데이터 흐름

```text
data/raw/{course_id}/{lecture_id}.pdf
   │
   ▼  lechal ingest {course_id} [--no-latex]
data/processed/{course_id}/
   ├── images/{lecture_id}/p001.png ...   # 페이지 이미지 (Split View 및 VLM 입력)
   ├── latex_cache/{sha256}.md            # 변환 결과 디스크 캐시
   ├── pages.jsonl                        # 페이지 메타데이터, 블록 bbox, 복잡도, latex_md
   └── chunks.jsonl                       # 페이지 단위 검색 텍스트 및 근거 bbox
   │
   ▼  lechal index {course_id} [--no-dense]
indexes/{course_id}/
   ├── bm25.json                          # 토큰 코퍼스 JSON 기반 BM25 인덱스
   └── chroma/                            # BGE-M3 + ChromaDB Dense 인덱스
   │
   ▼  lechal inspect {course_id} "{query}" [--dense]
상위 k개 근거 청크(chunk_id, score, 미리보기) 출력
```

### 복잡 페이지 판별 휴리스틱 (`complexity.py`)

모든 슬라이드를 VLM으로 변환하면 비용과 시간이 커지므로, PyMuPDF에서 추출한 4가지 지표를 기준으로 복잡 페이지만 선별한다.

1. 수학 유니코드 기호 개수 (`min_math_chars >= 3`)
2. 수식 폰트(`CMMI`, `CMSY`, `CambriaMath` 등) span 비율 (`math_font_span_ratio >= 0.1`)
3. 텍스트가 적은 페이지(`low_text_chars < 200`)에서 이미지 면적 비율 (`image_area_ratio >= 0.4`)
4. 벡터 도형 개수 (`min_drawings >= 40`)

---

## 4. 코드 리뷰 및 트러블슈팅

초기 M1 구현 이후 엣지 케이스 PDF(한글 파일명, 회전 페이지, 스캔 슬라이드, 빈 페이지, 겹치는 이미지, CropBox 등)와 코드 리뷰를 통해 발견하고 해결한 문제들이다.

### 문제 1. 짧은 과목 ID(`ml`)·한글 과목 ID(`확률론`)에서 ChromaDB 오류 발생

- **원인**: `DenseIndex`가 `course_id`를 그대로 ChromaDB `collection_name`으로 전달했다. ChromaDB는 컬렉션 이름에 `3~512자의 영숫자/._-` 제약을 강제하므로 `ml`이나 `확률론` 같은 과목 ID에서 `InvalidArgumentError`가 발생했다.
- **해결**: 이미 과목별로 저장 경로(`indexes/{course_id}/chroma`)가 물리적으로 분리되어 있으므로, 내부 컬렉션명은 고정값(`chunks`)을 사용하고 `course_id`는 컬렉션 메타데이터에 기록하도록 수정했다.

### 문제 2. 겹치는 배경 이미지가 있을 때 `image_ratio > 1.0` 오탐

- **원인**: `page.get_image_info()`로 얻은 이미지 면적을 단순 누적(`+=`)하다 보니, 슬라이드 배경 이미지나 워터마크가 겹치면 전체 페이지 면적을 초과(예: `image_ratio=1.12`)하는 현상이 발생했다.
- **해결**: 좌표 압축 기반 2D 스윕라인(`_rect_union_area`)으로 클리핑된 이미지 사각형들의 **실제 합집합 면적**을 계산하도록 변경해 항상 `0.0 <= image_ratio <= 1.0`을 보장했다.

### 문제 3. 스캔(이미지 전용) 슬라이드에서 `Chunk.bbox` 누락

- **원인**: `chunker.py`가 `kind == "text"` 블록의 bbox만 수집했기 때문에, 텍스트 레이어가 없는 스캔 슬라이드는 추후 LaTeX 변환으로 텍스트가 채워지더라도 `bbox=None`, `block_bboxes=[]`가 되는 문제가 있었다.
- **해결**: 텍스트 블록이 비어 있는 경우 이미지 블록(`kind == "image"`)의 `bbox`를 fallback으로 보존하도록 수정했다.

### 문제 4. API 키 미설정 시 모든 PDF 파싱이 끝난 뒤에야 실패 (Fail-late)

- **원인**: `ingest_course()`가 과목 내 모든 PDF를 파싱하고 PNG를 렌더링한 뒤에야 첫 LaTeX API 호출을 시도해 인증 오류(`LatexBackendUnavailable`)가 발생했다. 슬라이드가 수백 장이면 불필요한 작업 후 실패하게 된다.
- **해결**: `ClaudeVisionConverter.__init__` 단계에서 SDK의 인증 헤더(`auth_headers`) 및 토큰 캐시 유무를 선검사해, 키가 없으면 PDF 파싱 시작 전에 즉시 종료하고 `--no-latex` 안내 메시지를 출력하도록 개선했다.

### 문제 5. 캐시 정합성 및 파일 저장 안전성 보강

- **LaTeX 캐시 키**: 프롬프트에 텍스트 레이어(`raw_text`)가 힌트로 포함되므로, `cache_key` 해시 입력에 `raw_text`도 추가해 텍스트 파싱 변경 시 캐시가 자동 무효화되게 했다.
- **잔존 PNG 정리**: 동일 강의 PDF의 페이지 수가 줄어든 상태로 재파싱할 때 기존 `p0xx.png` 파일이 남지 않도록 정리 로직을 추가했다.
- **원자적 저장**: `bm25.json` 저장 시에도 `store.write_jsonl`과 동일하게 `.tmp` 파일을 먼저 쓰고 `replace()`로 교체하도록 변경했다.
- **PyMuPDF 2.x 호환**: `import fitz`가 deprecated됨에 따라 `import pymupdf`로 통일했다.

---

## 5. 실제 강의 PDF 검증 및 테스트 결과

### 자동화 테스트 (`pytest`)

초기 25개 테스트에 CLI 전체 흐름, API 키 선검사, 겹치는 이미지 합집합, 회전/빈 페이지, 잔존 PNG 정리, 이미지 전용 청크, ChromaDB 한글/짧은 과목 ID 테스트를 추가해 **총 36개 테스트 전체 통과**를 확인했다.

```bash
uv run pytest -v
# 36 passed
```

### 실제 강의자료(`dg`, 미분기하개론) 적용 결과

실제 수학 강의 PDF 2개(`lec61.pdf`, `lec62.pdf`, 총 5페이지)에 `--no-latex` 옵션으로 파이프라인을 실행해 보았다.

- **파싱 및 복잡도 감지**: 5페이지 모두 빈 텍스트 없이 추출되었고, `math_chars`와 `math_font_ratio` 기준으로 5/5 페이지가 복잡 페이지로 정확히 감지되었다.
- **관찰된 한계 (TeX PDF 텍스트 레이어)**: 본문, 정리/정의 제목, 인라인 수식(`ωij(v) = (∇vEi) · Ej(p)` 등)은 M2 검색 개발에 충분할 정도로 깔끔하게 추출되었으나, TeX PDF 특성상 `\sum` 같은 대형 연산자가 `X`와 분리된 줄바꿈(`3 X ωij(v)Ej(p) . ∇vEi = j=1`)으로 깨지는 현상이 관찰되었다.
- **대응 방향**: 당장은 API 비용 없이 `--no-latex` 기반으로 M2 파이프라인을 우선 구축하고, M3 정량 평가 단계에서 LaTeX 변환 백엔드(Claude Vision 또는 로컬 VLM/OCR)를 켜서 수식 Claim 검증 정확도 차이를 비교할 계획이다.

---

## 6. 다음 계획 (Next Steps)

M1 구현과 리뷰 개선 사항을 모두 `main` 브랜치에 병합 완료했다. 다음 단계(M2)에서는 아래 기능을 구현할 예정이다.

- [ ] **Hybrid Retrieval**: BM25 + Dense 검색 결과의 RRF(Reciprocal Rank Fusion) 결합
- [ ] **Reranking**: Cross-Encoder Reranker로 상위 근거 슬라이드 정제
- [ ] **Claim Decomposition & NLI Verification**: 답변을 원자적 Claim으로 분해하고 NLI 모델로 할루시네이션 여부를 판별하는 CLI 구현
- [ ] **(선택)** CUDA PyTorch 환경 구성으로 BGE-M3 및 NLI 추론 GPU 가속
