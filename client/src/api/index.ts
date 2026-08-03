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
    type: 'token' | 'done' | 'error' | 'stopped';
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
 * Pass `signal` (from an AbortController) to let the caller cancel client-side
 * reading — call stopChatStream() as well to actually stop generation on the
 * server, since aborting the fetch alone doesn't cancel the backend's task.
 */
export async function sendChatMessageStream(
    message: string,
    sessionId: string | undefined,
    onToken: (token: string) => void,
    onDone: (metadata: StreamEvent) => void,
    onError?: (error: string) => void,
    signal?: AbortSignal,
): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ message, session_id: sessionId }),
        signal,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to send message' }));
        throw new Error(error.error || error.detail || 'Failed to send message');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    try {
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
                    } else if (event.type === 'done' || event.type === 'stopped') {
                        onDone(event);
                    } else if (event.type === 'error') {
                        onError?.(event.content || 'Unknown streaming error');
                    }
                } catch {
                    // Skip malformed JSON lines
                }
            }
        }
    } catch (e: any) {
        // The user clicking Stop aborts the fetch — that's an intentional
        // stop, not a failure, so don't surface it as an error.
        if (e?.name === 'AbortError') return;
        throw e;
    }
}

/**
 * Stop button: tells the server to cancel the in-flight generation for this
 * session (see /api/chat/stream/stop). Call alongside aborting the fetch —
 * closing the client's connection alone does not stop server-side generation.
 */
export async function stopChatStream(sessionId: string): Promise<void> {
    await fetch(`${API_BASE_URL}/chat/stream/stop`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
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
    if (!response.ok) {
        throw new Error('Failed to check chat health');
    }
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

// ============================================================================
// Bookings API (Braintree sandbox + Postgres)
// ============================================================================

export interface Booking {
    id: string;
    userId: string;
    lawyerId: string;
    amount: number;
    transactionId: string;
    status: string;
    appointmentDate?: string | null;
    appointmentTime?: string | null;
    createdAt?: string | null;
}

/** One-time Braintree client token authorizing the drop-in UI (plain text). */
export async function fetchBraintreeClientToken(): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/bookings/client_token`);
    if (!response.ok) {
        throw new Error('Failed to get payment token');
    }
    return response.text();
}

export interface CheckoutPayload {
    amount: string;
    paymentMethodNonce: string;
    lawyerId: string;
    userId: string;
}

/** Charge the nonce and record the confirmed booking. */
export async function checkoutBooking(payload: CheckoutPayload) {
    const response = await fetch(`${API_BASE_URL}/bookings/checkout`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...getAuthHeaders(),
        },
        body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status === 'error') {
        throw new Error(data.message || 'Payment failed');
    }
    return data as { status: string; transactionId: string };
}

/** Confirmed appointments for a user, newest first. */
export async function fetchUserBookings(userId: string): Promise<Booking[]> {
    const response = await fetch(`${API_BASE_URL}/bookings/user-bookings/${userId}`, {
        headers: { ...getAuthHeaders() },
    });
    if (!response.ok) {
        throw new Error('Failed to fetch bookings');
    }
    return response.json();
}

// ============================================================================
// Shared error helper for the new-feature endpoints below, which return
// {"message": ...} bodies (see server/CLAUDE.md's API compatibility rules).
// ============================================================================

async function requestJson(path: string, options: RequestInit = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        headers: { ...(options.headers || {}), ...getAuthHeaders() },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.message || 'Request failed');
    }
    return data;
}

// ============================================================================
// Bare Act Explorer
// ============================================================================

export async function searchBareAct(query: string, actHint?: string) {
    const params = new URLSearchParams({ q: query });
    if (actHint) params.set('act', actHint);
    return requestJson(`/bare-acts/search?${params.toString()}`);
}

// ============================================================================
// Similar Case Search
// ============================================================================

export async function searchSimilarCases(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    return requestJson('/similar-cases/search', { method: 'POST', body: formData });
}

export async function searchSimilarCasesByText(text: string) {
    return requestJson('/similar-cases/search-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });
}

// ============================================================================
// My Cases
// ============================================================================

export interface SavedCase {
    id: string;
    cnr: string | null;
    court: string | null;
    caseNumber: string | null;
    year: number | null;
    title: string | null;
    status: string | null;
    lastSyncedAt: string | null;
    createdAt: string | null;
}

export async function saveCase(payload: { cnr?: string; court?: string; caseNumber?: string; year?: number }) {
    return requestJson('/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
}

export async function fetchSavedCases(): Promise<SavedCase[]> {
    return requestJson('/cases');
}

export async function fetchCaseDetail(caseId: string) {
    return requestJson(`/cases/${caseId}`);
}

export async function addCaseNote(caseId: string, noteText: string) {
    return requestJson(`/cases/${caseId}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ noteText }),
    });
}

export async function syncCase(caseId: string) {
    return requestJson(`/cases/${caseId}/sync`, { method: 'POST' });
}

export async function deleteCase(caseId: string) {
    return requestJson(`/cases/${caseId}`, { method: 'DELETE' });
}

// ============================================================================
// Court Cause List Search
// ============================================================================

export async function searchCauseList(params: { court: string; date: string; advocate?: string; judge?: string; caseNumber?: string }) {
    const qs = new URLSearchParams({ court: params.court, date: params.date });
    if (params.advocate) qs.set('advocate', params.advocate);
    if (params.judge) qs.set('judge', params.judge);
    if (params.caseNumber) qs.set('caseNumber', params.caseNumber);
    return requestJson(`/cause-list/search?${qs.toString()}`);
}

// ============================================================================
// Legal Document Vault
// ============================================================================

export interface VaultDocument {
    id: string;
    title: string;
    documentType: string;
    fileSizeBytes: number;
    mimeType: string;
    relatedCaseId: string | null;
    indexingStatus: string;
    createdAt: string | null;
}

export async function uploadVaultDocument(file: File, title: string, documentType: string, relatedCaseId?: string) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('documentType', documentType);
    if (relatedCaseId) formData.append('relatedCaseId', relatedCaseId);
    return requestJson('/vault/documents', { method: 'POST', body: formData });
}

export async function fetchVaultDocuments(): Promise<VaultDocument[]> {
    return requestJson('/vault/documents');
}

export async function searchVaultDocuments(query: string) {
    return requestJson('/vault/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });
}

export async function deleteVaultDocument(documentId: string) {
    return requestJson(`/vault/documents/${documentId}`, { method: 'DELETE' });
}

// ============================================================================
// Personal Legal Calendar
// ============================================================================

export interface CalendarEvent {
    id: string;
    title: string;
    eventType: string;
    startAt: string;
    endAt: string | null;
    relatedCaseId: string | null;
}

export async function fetchCalendarEvents(): Promise<CalendarEvent[]> {
    return requestJson('/calendar/events');
}

export async function createCalendarEvent(payload: { title: string; eventType: string; startAt: string; endAt?: string }) {
    return requestJson('/calendar/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
}

export async function deleteCalendarEvent(eventId: string) {
    return requestJson(`/calendar/events/${eventId}`, { method: 'DELETE' });
}

// ============================================================================
// Smart Notifications
// ============================================================================

export interface AppNotification {
    id: string;
    type: string;
    title: string;
    body: string;
    channel: string;
    status: string;
    readAt: string | null;
    createdAt: string | null;
}

export async function fetchNotifications(): Promise<AppNotification[]> {
    return requestJson('/notifications');
}

export async function markNotificationRead(notificationId: string) {
    return requestJson(`/notifications/${notificationId}/read`, { method: 'PATCH' });
}
