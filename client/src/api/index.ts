import { API_BASE_URL } from './config';

const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export interface LoginCredentials {
    email: string;
    password: string;
}

export interface RegisterData {
    name: string;
    email: string;
    password: string;
}

export interface LawyerCriteria {
    specialty?: string;
    location?: string;
    [key: string]: any;
}

// ============================================================================
// Chat API - Connected to Python Chatbot via Express proxy
// ============================================================================

export interface ChatResponse {
    response: string;
    session_id: string;
    intent?: string;
    document_info?: Record<string, any>;
    crime_report?: Record<string, any>;
    lawyers_found?: Array<Record<string, any>>;
}

export interface ChatMessage {
    message: string;
    session_id?: string;
}

/**
 * Send a chat message to the AI legal assistant
 */
export async function sendChatMessage(message: string, sessionId?: string): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ message, session_id: sessionId }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to send message' }));
        throw new Error(error.error || error.detail || 'Failed to send message');
    }
    return response.json();
}

export interface StreamEvent {
    type: 'token' | 'done' | 'error';
    content?: string;
    session_id?: string;
    intent?: string;
    lawyers_found?: Array<Record<string, any>>;
    document_info?: Record<string, any>;
    document_validation?: Record<string, any>;
    crime_report?: Record<string, any>;
}

/**
 * Send a chat message and stream the response token by token via SSE.
 * Calls `onToken` for each LLM token and `onDone` with metadata when complete.
 */
export async function sendChatMessageStream(
    message: string,
    sessionId: string | undefined,
    onToken: (token: string) => void,
    onDone: (metadata: StreamEvent) => void,
    onError?: (error: string) => void,
): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to send message' }));
        throw new Error(error.error || error.detail || 'Failed to send message');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;

            const jsonStr = trimmed.slice(6); // Remove "data: " prefix
            try {
                const event: StreamEvent = JSON.parse(jsonStr);

                if (event.type === 'token' && event.content) {
                    onToken(event.content);
                } else if (event.type === 'done') {
                    onDone(event);
                } else if (event.type === 'error') {
                    onError?.(event.content || 'Unknown streaming error');
                }
            } catch {
                // Skip malformed JSON lines
            }
        }
    }
}

/**
 * Upload a document for AI analysis
 */
export async function uploadDocumentForAnalysis(
    file: File,
    message?: string,
    sessionId?: string
): Promise<ChatResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (message) formData.append('message', message);
    if (sessionId) formData.append('session_id', sessionId);

    const response = await fetch(`${API_BASE_URL}/chat/upload`, {
        method: 'POST',
        headers: {
            ...getAuthHeaders(),
        },
        body: formData,
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to upload document' }));
        throw new Error(error.error || error.detail || 'Failed to upload document');
    }
    return response.json();
}

/**
 * Analyze document text directly without file upload
 */
export async function analyzeDocumentText(
    documentText: string,
    sessionId?: string
): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat/analyze-document`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ document_text: documentText, session_id: sessionId }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to analyze document' }));
        throw new Error(error.error || error.detail || 'Failed to analyze document');
    }
    return response.json();
}

/**
 * Get crime reporting guidance
 */
export async function getCrimeReportGuidance(
    description: string,
    sessionId?: string
): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat/crime-report`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ description, session_id: sessionId }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to get guidance' }));
        throw new Error(error.error || error.detail || 'Failed to get crime report guidance');
    }
    return response.json();
}

/**
 * Find lawyers using AI-powered search
 */
export async function findLawyersAI(
    query: string,
    location?: string,
    specialization?: string
) {
    const response = await fetch(`${API_BASE_URL}/chat/find-lawyer`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ query, location, specialization }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to find lawyers' }));
        throw new Error(error.error || error.detail || 'Failed to find lawyers');
    }
    return response.json();
}

/**
 * Get available legal specializations
 */
export async function getSpecializations(): Promise<{ specializations: string[] }> {
    const response = await fetch(`${API_BASE_URL}/chat/specializations`);
    if (!response.ok) {
        throw new Error('Failed to fetch specializations');
    }
    return response.json();
}

/**
 * Get recognized crime types
 */
export async function getCrimeTypes(): Promise<{ crime_types: string[] }> {
    const response = await fetch(`${API_BASE_URL}/chat/crime-types`);
    if (!response.ok) {
        throw new Error('Failed to fetch crime types');
    }
    return response.json();
}

/**
 * Clear a chat session
 */
export async function clearChatSession(sessionId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/chat/session/${sessionId}`, {
        method: 'DELETE',
        headers: {
            ...getAuthHeaders(),
        },
    });
    if (!response.ok) {
        throw new Error('Failed to clear session');
    }
    return response.json();
}

/**
 * Get chat session history
 */
export async function getChatSessionHistory(sessionId: string) {
    const response = await fetch(`${API_BASE_URL}/chat/session/${sessionId}/history`, {
        headers: {
            ...getAuthHeaders(),
        },
    });
    if (!response.ok) {
        throw new Error('Failed to fetch session history');
    }
    return response.json();
}

/**
 * Check chatbot service health
 */
export async function checkChatHealth() {
    const response = await fetch(`${API_BASE_URL}/chat/health`);
    return response.json();
}

// ============================================================================
// Lawyers API
// ============================================================================

export async function fetchLawyers() {
    const response = await fetch(`${API_BASE_URL}/lawyers`);
    if (!response.ok) {
        throw new Error('Failed to fetch lawyers');
    }
    return response.json();
}

export async function fetchLawyerById(id: string) {
    const response = await fetch(`${API_BASE_URL}/lawyers/${id}`);
    if (!response.ok) {
        throw new Error('Failed to fetch lawyer');
    }
    return response.json();
}

export async function recommendLawyers(criteria: LawyerCriteria) {
    const response = await fetch(`${API_BASE_URL}/lawyers/recommend`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(criteria),
    });
    if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
    }
    return response.json();
}

// ============================================================================
// Authentication API
// ============================================================================

export async function login(credentials: LoginCredentials) {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Login failed');
    }
    return response.json();
}

export async function register(userData: RegisterData) {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Registration failed');
    }
    return response.json();
}

export async function fetchUserProfile() {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
            ...getAuthHeaders(),
        },
    });
    if (!response.ok) {
        throw new Error('Failed to fetch profile');
    }
    return response.json();
}
