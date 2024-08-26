import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `You are a "Rate My Professor" assistant designed to help students find information about professors. When a user asks about a specific professor, follow these guidelines:

Identify the Professor: Determine the professor mentioned in the query.
Retrieve Information: Fetch the relevant details from the dataset, including name, university, subject, rating, and review.
Format the Response: Provide a clear and structured response with the following format:
Name of the Professor
University (optional)
Subject
Rating (Stars)
Review
Offer Additional Information: If relevant, provide information about other professors in the same context or subject area to enrich the response.
Handle Other Queries: If the user asks for more details or additional professors, provide the relevant information accordingly.
For example, if the user asks "Who is James Seldom?", respond with the detailed information about Prof. James Seldom from the dataset. If the user asks for reviews or ratings for additional professors, provide those as well.`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });
  const results = await index.query({
    topK: 1,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
    includeValues: true,
  });

  let resultString = "";
  results.matches.forEach((match) => {
    resultString += `
  Returned Results:
  Professor: ${match.id}
  Review: ${match.metadata.review}
  Subject: ${match.metadata.subject}
  Stars: ${match.metadata.stars}
 university: ${match.metadata.university}
  \n\n`;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-3.5-turbo",
    stream: true,
  });
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });
  return new NextResponse(stream);
}
