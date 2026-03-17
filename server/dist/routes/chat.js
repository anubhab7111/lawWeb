"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const router = express_1.default.Router();
// Initialize OpenAI (requires API key in .env)
// const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
router.post('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { message } = req.body;
    try {
        // Placeholder logic for now since we don't have API Key yet
        // In production:
        // const completion = await openai.chat.completions.create({
        //   messages: [{ role: "system", content: "You are a legal assistant." }, { role: "user", content: message }],
        //   model: "gpt-3.5-turbo",
        // });
        // const reply = completion.choices[0].message.content;
        // Simple mock response
        const reply = `This is a mock response from the backend for: "${message}". Please configure OpenAI API key to get real responses.`;
        res.json({ reply });
    }
    catch (err) {
        res.status(500).json({ message: err.message });
    }
}));
exports.default = router;
