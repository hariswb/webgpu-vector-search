import { test } from "vitest";
import { VectorSearchPipeline } from "../pipeline";

const pipeline = new VectorSearchPipeline("https://hariswb.github.io/indonesian-news-2024-2025",10)
await pipeline.init()

test('Pipeline 0', async ()=>{
    const result = await pipeline.compute("Krisis ekonomi")
    console.log("Result",result)
})

test('Pipeline 1', async ()=>{
    const result = await pipeline.compute("Demonstrasi ricuh")
    console.log("Result",result)
})
