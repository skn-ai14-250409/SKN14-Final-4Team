from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore.from_existing_index(
    index_name="transcripts-korean", 
    embedding=embeddings,
    namespace="transcripts-kr" 
)

index=vector_store.index
stats = index.describe_index_stats()
print(stats)

# 벡터 저장소 검색기 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# 프롬프트 템플릿 생성
from langchain_core.output_parsers import StrOutputParser

template = """
당신은 재활용소재로 패션제품을 만드는 회사의 고문이며 친환경 활동가이자 패션/스타일링 전문가다.
<<스타일링 팁>>을 "가능하면" 활용해 사용자가 원하는 스타일 3개를 추천하고, 각 스타일마다 상의/하의 조합을 3개씩 추천한다.
규칙:
1) 인사말/자기소개/군더더기 금지
2) 상의/하의 조합이 없는 스타일은 제외

[Context]
{context}

[Question]
{query}

[Answer]
"""

prompt = ChatPromptTemplate.from_template(template)
qa_chain = prompt | llm | StrOutputParser()

# 문서 포맷팅 함수
def format_docs(relevant_docs):
    """검색된 문서들을 하나의 문자열로 결합한다"""
    return "\n".join(doc.page_content for doc in relevant_docs)


import pandas as pd

dataset_df = pd.read_csv('/Users/yun-iseo/Workspaces/SKN14-Final-4Team/LLM_evaluation/ragas_kg_dataset.csv')
print("샘플 데이터:")
print(dataset_df.head(3))

eval_dataset = dataset_df[['user_input', 'reference_contexts', 'reference']]

# 평가용 데이터셋 생성
evaluated_dataset = []

# 각 행에 대해 RAG 체인을 호출하여 결과를 저장
for row in eval_dataset.itertuples():
    query = row.user_input  # 사용자 입력
    retrieved_contexts = retriever.invoke(query)  # 실제 검색된 문서
    relevant_docs = retriever.invoke(query)
    response = qa_chain.invoke(  # RAG 체인으로 답변 생성
        {
            "context": format_docs(relevant_docs),
            "query": query,
        }
    )

    reference = row.reference  # 정답
    reference_contexts = row.reference_contexts  # 정답 참조 컨텍스트

    evaluated_dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
            # "reference_contexts": reference_contexts,
        }
    )

# RAGAS 평가 데이터셋 생성
ragas_evaluated_dataset = EvaluationDataset.from_list(evaluated_dataset)

# 데이터 저장
ragas_evaluated_dataset.to_pandas().to_csv('ragas_evaluated_dataset.csv', index=False)