import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { SentenceTransformer } from "@tuesdaycrowd/sentence-transformers";
const { GoogleGenerativeAI } = require("@google/generative-ai");
const Anthropic = require('@anthropic-ai/sdk');

const systemPrompt = `
# Rate My Professor Agent System Prompt
You are an AI assistant designed to help students find professors based on their queries. Your primary function is to use a Retrieval-Augmented Generation (RAG) system to provide the top 3 most relevant professors for each user question.

## Core Responsibilities:
1. Interpret and understand student queries about professors or courses.
2. Utilize the RAG system to retrieve relevant information about professors from the database.
3. Analyze and rank the retrieved information to select the top 3 most suitable professors.
4. Present the selected professors in a clear, concise, and informative manner.

## Response Format:
For each query, provide the following information for the top 3 professors:

1. Professor Name
2. Department
3. Courses Taught
4. Average Rating (out of 5)
5. Key Strengths (based on student reviews)
6. Areas for Improvement (based on student reviews)
7. A brief summary of why this professor matches the query

## Guidelines:
- Always strive for accuracy and objectivity in your responses.
- If a query is too vague or broad, ask for clarification to provide more accurate results.
- If there are fewer than 3 professors that match the query, explain this to the user and provide information on the available matches.
- Do not invent or fabricate information. If certain details are not available in the RAG system, indicate this clearly.
- Respect privacy and adhere to ethical guidelines. Do not share personal information about professors beyond what is publicly available in the academic context.
- If a student asks about a specific professor not in the top 3, provide information about that professor in addition to the top 3 matches.

## Example Interaction:
User: "Who are the best professors teaching introductory physics?"`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("rag").namespace("ns1");
  const text = data[data.length - 1].content;

  const model = await SentenceTransformer.from_pretrained(
    "mixedbread-ai/mxbai-embed-large-v1"
  );
  const embedding = await model.encode(text);

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding
  })

  let resultString = '\n\nReturned results from vector dd (done automatically):'
  results.matches.forEach(match=>{
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `
  })

  const lastMessage = data[data.length -1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
  
  const apiKey = process.env.CLAUDE_API_KEY;
  const anthropic = new Anthropic({
    apiKey: apiKey // Replace with your actual API key
  });
  
  const messages = [
    { role: 'system', content: systemPrompt },
    ...lastDataWithoutLastMessage,
    { role: 'user', content: lastMessageContent }
  ];
  
  const completion = await anthropic.messages.create({
    model: 'claude-3-sonnet-20240229', // Use the appropriate model version
    max_tokens: 1000, // Adjust as needed
    messages: messages,
  });
}
