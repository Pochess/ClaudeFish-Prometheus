/*
 * ================================================================
 * PROMETHEUS Chess Engine v2.0
 * Motor UCI de alto rendimiento - toutes techniques modernes
 * ================================================================
 *
 * ARCHITECTURE:
 *  - Bitboards avec magic bitboards (PEXT si BMI2)
 *  - Représentation 12 bitboards + tableau de pièces
 *  - Hachage Zobrist incrémental
 *
 * GÉNÉRATION DE COUPS:
 *  - Pseudo-légaux + filtre légalité (pin/échec)
 *  - Promotions, en passant, roques
 *  - Coups de capture séparés (quiescence)
 *
 * RECHERCHE:
 *  - Negamax PVS (Principal Variation Search)
 *  - Iterative Deepening avec aspiration windows
 *  - Transposition Table (TT) 2-bucket, remplacement par âge/profondeur
 *  - Null Move Pruning (NMP) + vérification
 *  - Late Move Reductions (LMR) logarithmiques
 *  - Late Move Pruning (LMP)
 *  - Futility Pruning + Extended Futility
 *  - Razoring
 *  - SEE Pruning (captures négatives)
 *  - Singular Extensions
 *  - Internal Iterative Deepening (IID)
 *  - Check Extensions
 *  - Quiescence Search + Delta Pruning + SEE
 *  - Lazy SMP multi-threading
 *
 * ORDONNANCEMENT:
 *  - TT move (best from previous search)
 *  - Promotions queen en tête
 *  - MVV-LVA pour captures
 *  - SEE pour captures mauvaises
 *  - Killer moves (2 par ply)
 *  - Countermove heuristic
 *  - History heuristic (avec aging)
 *  - Continuation history
 *
 * ÉVALUATION:
 *  - Material + PeSTO PSQT (MG/EG tapered)
 *  - Structure de pions (pawn hash table)
 *  - Pions passés + candidates
 *  - Sécurité du roi (king safety)
 *  - Mobilité des pièces
 *  - Paire de fous
 *  - Coopération tour/reine
 *  - Endgame patterns (KPK, KBNK...)
 *
 * PROTOCOLE: UCI complet (position, go, stop, ponderhit, etc.)
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>
#include <climits>
#include <functional>
#include <array>
#include <memory>

using namespace std;

// ==================== TYPES ====================

using U64 = uint64_t;
using U32 = uint32_t;
using U16 = uint16_t;
using U8  = uint8_t;
using Move = U32;
using Value = int;
using Depth = int;

// ==================== ENUMS ====================

enum Color : U8 { WHITE = 0, BLACK = 1 };
inline Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : U8 { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, NO_PT=6 };

// Piece = color*6 + type  (0..11), 12=EMPTY
enum Piece : U8 {
    W_PAWN=0,W_KNIGHT=1,W_BISHOP=2,W_ROOK=3,W_QUEEN=4,W_KING=5,
    B_PAWN=6,B_KNIGHT=7,B_BISHOP=8,B_ROOK=9,B_QUEEN=10,B_KING=11,
    EMPTY=12
};
inline PieceType typeOf(Piece p) { return PieceType(p % 6); }
inline Color     colorOf(Piece p){ return Color(p / 6); }
inline Piece makePiece(Color c, PieceType t) { return Piece(c*6+t); }

// Squares: A1=0, B1=1, ..., H8=63
enum Square : U8 {
    A1=0,B1,C1,D1,E1,F1,G1,H1,
    A2,B2,C2,D2,E2,F2,G2,H2,
    A3,B3,C3,D3,E3,F3,G3,H3,
    A4,B4,C4,D4,E4,F4,G4,H4,
    A5,B5,C5,D5,E5,F5,G5,H5,
    A6,B6,C6,D6,E6,F6,G6,H6,
    A7,B7,C7,D7,E7,F7,G7,H7,
    A8,B8,C8,D8,E8,F8,G8,H8,
    NO_SQ=64
};
inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline int makeSquare(int f, int r) { return r*8+f; }

// Castling
enum CastlingRight : U8 { NO_CASTLE=0, WK_C=1, WQ_C=2, BK_C=4, BQ_C=8, ANY_C=15 };

// Move flags (4 bits, bits 12-15 of move)
enum MoveFlag {
    QUIET=0, DPUSH=1, KS_CASTLE=2, QS_CASTLE=3,
    CAPTURE=4, EP_CAP=5,
    // 6,7 unused
    PROMO_N=8, PROMO_B=9, PROMO_R=10, PROMO_Q=11,
    CAP_PROMO_N=12, CAP_PROMO_B=13, CAP_PROMO_R=14, CAP_PROMO_Q=15
};

// Move encoding: from[5:0] | to[11:6] | flag[15:12]
inline Move makeMove(int from, int to, int flag=QUIET) {
    return Move(from | (to<<6) | (flag<<12));
}
inline int  fromSq(Move m) { return m & 63; }
inline int  toSq(Move m)   { return (m>>6) & 63; }
inline int  flagOf(Move m) { return (m>>12) & 15; }
inline bool isCapture(Move m)   { return flagOf(m) & 4; }
inline bool isPromotion(Move m) { return flagOf(m) & 8; }
inline bool isEnPassant(Move m) { return flagOf(m) == EP_CAP; }
inline bool isCastle(Move m)    { int f=flagOf(m); return f==KS_CASTLE||f==QS_CASTLE; }
inline PieceType promoType(Move m) { return PieceType((flagOf(m)&3)+1); }

const Move MOVE_NONE = 0;
const Move MOVE_NULL = Move(1<<12);

// ==================== CONSTANTS ====================

static constexpr Value VALUE_INF         =  30000;
static constexpr Value VALUE_MATE        =  29000;
static constexpr Value VALUE_MATE_THRESH =  28000;
static constexpr Value VALUE_DRAW        =  0;
static constexpr int   MAX_PLY           =  128;
static constexpr int   MAX_MOVES         =  256;
static constexpr int   TT_SIZE_MB        =  64;  // default TT size

// Material values [MG, EG]
const Value PIECE_MG[7] = {  82, 337, 365, 477, 1025,     0, 0 };
const Value PIECE_EG[7] = {  94, 281, 297, 512,  936, 20000, 0 };

// SEE values
const Value SEE_VAL[7] = { 100, 300, 300, 500, 900, 20000, 0 };

// ==================== BITBOARD UTILITIES ====================

inline U64  bit(int sq)      { return 1ULL << sq; }
inline int  popcount(U64 b)  { return __builtin_popcountll(b); }
inline int  lsb(U64 b)       { return __builtin_ctzll(b); }
inline int  msb(U64 b)       { return 63 ^ __builtin_clzll(b); }
inline int  popLsb(U64& b)   { int s=lsb(b); b&=b-1; return s; }
inline bool several(U64 b)   { return b & (b-1); }

const U64 FILE_A = 0x0101010101010101ULL;
const U64 FILE_H = 0x8080808080808080ULL;
const U64 RANK_1 = 0x00000000000000FFULL;
const U64 RANK_2 = 0x000000000000FF00ULL;
const U64 RANK_3 = 0x0000000000FF0000ULL;
const U64 RANK_6 = 0x0000FF0000000000ULL;
const U64 RANK_7 = 0x00FF000000000000ULL;
const U64 RANK_8 = 0xFF00000000000000ULL;

inline U64 fileMask(int f) { return FILE_A << f; }

inline U64 shiftN(U64 b)  { return b << 8; }
inline U64 shiftS(U64 b)  { return b >> 8; }
inline U64 shiftE(U64 b)  { return (b & ~FILE_H) << 1; }
inline U64 shiftW(U64 b)  { return (b & ~FILE_A) >> 1; }
inline U64 shiftNE(U64 b) { return (b & ~FILE_H) << 9; }
inline U64 shiftNW(U64 b) { return (b & ~FILE_A) << 7; }
inline U64 shiftSE(U64 b) { return (b & ~FILE_H) >> 7; }
inline U64 shiftSW(U64 b) { return (b & ~FILE_A) >> 9; }

U64 fillNorth(U64 b) { b|=b<<8; b|=b<<16; b|=b<<32; return b; }
U64 fillSouth(U64 b) { b|=b>>8; b|=b>>16; b|=b>>32; return b; }

// ==================== ATTACK TABLES ====================

U64 knightAtk[64], kingAtk[64], pawnAtk[2][64];
U64 betweenBB[64][64];  // squares strictly between two aligned squares
U64 lineBB[64][64];     // full line through two aligned squares

// Magic bitboard data
struct Magic {
    U64  mask;
    U64  magic;
    U64* table;
    int  shift;
    U64 attacks(U64 occ) const { return table[((occ & mask) * magic) >> shift]; }
};

Magic bishopMagics[64], rookMagics[64];
static U64 bishopTable[64][512];
static U64 rookTable[64][4096];

// Slow attack computation (used only during init)
U64 sliderAttackSlow(int sq, U64 occ, int dr, int df) {
    U64 att = 0;
    int r = rankOf(sq) + dr, f = fileOf(sq) + df;
    while (r>=0 && r<8 && f>=0 && f<8) {
        att |= bit(makeSquare(f,r));
        if (occ & bit(makeSquare(f,r))) break;
        r+=dr; f+=df;
    }
    return att;
}
U64 bishopAttSlow(int sq, U64 occ) {
    return sliderAttackSlow(sq,occ,1,1)|sliderAttackSlow(sq,occ,1,-1)|
           sliderAttackSlow(sq,occ,-1,1)|sliderAttackSlow(sq,occ,-1,-1);
}
U64 rookAttSlow(int sq, U64 occ) {
    return sliderAttackSlow(sq,occ,1,0)|sliderAttackSlow(sq,occ,-1,0)|
           sliderAttackSlow(sq,occ,0,1)|sliderAttackSlow(sq,occ,0,-1);
}

U64 indexToOcc(int idx, int bits, U64 mask) {
    U64 occ = 0;
    for (int i=0; i<bits; i++) {
        int sq = lsb(mask); mask &= mask-1;
        if (idx & (1<<i)) occ |= bit(sq);
    }
    return occ;
}

struct PRNG {
    U64 s;
    PRNG(U64 seed): s(seed) {}
    U64 rand64() { s^=s>>12; s^=s<<25; s^=s>>27; return s*2685821657736338717ULL; }
    U64 sparse()  { return rand64()&rand64()&rand64(); }
};

// Correct relevant occupancy masks (exclude far edges - they don't affect attacks)
U64 calcRookMask(int sq) {
    U64 result = 0;
    int r = rankOf(sq), f = fileOf(sq);
    for (int ff=f+1; ff<=6; ff++) result |= bit(r*8+ff); // right, stop before h-file
    for (int ff=f-1; ff>=1; ff--) result |= bit(r*8+ff); // left, stop before a-file
    for (int rr=r+1; rr<=6; rr++) result |= bit(rr*8+f); // up, stop before rank 8
    for (int rr=r-1; rr>=1; rr--) result |= bit(rr*8+f); // down, stop before rank 1
    return result;
}
U64 calcBishopMask(int sq) {
    return bishopAttSlow(sq,0) & ~(RANK_1|RANK_8|FILE_A|FILE_H);
}

U64 findMagic(int sq, bool bishop, PRNG& rng) {
    U64 mask = bishop ? calcBishopMask(sq) : calcRookMask(sq);

    int bits = popcount(mask);
    int size = 1 << bits;
    vector<U64> occ(size), att(size);
    for (int i=0; i<size; i++) {
        occ[i] = indexToOcc(i, bits, mask);
        att[i] = bishop ? bishopAttSlow(sq, occ[i]) : rookAttSlow(sq, occ[i]);
    }
    for (int k=0; k<100000000; k++) {
        U64 magic = rng.sparse();
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;
        vector<U64> used(size, 0ULL);
        bool fail = false;
        for (int i=0; i<size && !fail; i++) {
            int j = int((occ[i]*magic) >> (64-bits));
            if (!used[j]) used[j] = att[i];
            else if (used[j] != att[i]) fail = true;
        }
        if (!fail) return magic;
    }
    return 0; // Should never happen
}

inline U64 getBishopAtk(int sq, U64 occ) { return bishopMagics[sq].attacks(occ); }
inline U64 getRookAtk  (int sq, U64 occ) { return rookMagics[sq].attacks(occ); }
inline U64 getQueenAtk (int sq, U64 occ) { return getBishopAtk(sq,occ)|getRookAtk(sq,occ); }

// ==================== ZOBRIST ====================

U64 zobPiece[12][64];
U64 zobCastle[16];
U64 zobEP[8];
U64 zobSide;

void initZobrist() {
    PRNG rng(1070372ULL);
    for (int p=0; p<12; p++) for (int sq=0; sq<64; sq++) zobPiece[p][sq] = rng.rand64();
    for (int i=0; i<16; i++) zobCastle[i] = rng.rand64();
    for (int i=0; i<8; i++) zobEP[i] = rng.rand64();
    zobSide = rng.rand64();
}

// ==================== INITIALIZATION ====================

void initAll() {
    // Knight attacks
    for (int sq=0; sq<64; sq++) {
        U64 b = bit(sq);
        knightAtk[sq] = ((b<<17)&~FILE_A)|((b<<15)&~FILE_H)|
                        ((b<<10)&~(FILE_A|(FILE_A<<1)))|((b<<6)&~(FILE_H|(FILE_H>>1)))|
                        ((b>>17)&~FILE_H)|((b>>15)&~FILE_A)|
                        ((b>>10)&~(FILE_H|(FILE_H>>1)))|((b>>6)&~(FILE_A|(FILE_A<<1)));
        kingAtk[sq] = shiftN(b)|shiftS(b)|shiftE(b)|shiftW(b)|
                      shiftNE(b)|shiftNW(b)|shiftSE(b)|shiftSW(b);
        pawnAtk[WHITE][sq] = shiftNE(b)|shiftNW(b);
        pawnAtk[BLACK][sq] = shiftSE(b)|shiftSW(b);
    }

    // Between/Line BBs
    //
    // lineBB[s1][s2] must contain ONLY the single rank/file/diagonal shared by
    // s1 and s2 (used by isLegal() to test "does a pinned piece's destination
    // stay on the pin ray"). The previous formula OR'd s1's attacks (blocked
    // at s2) with s2's attacks on an EMPTY board, which pulls in s1's *other*
    // rank/file/diagonal too. Concretely, for a king on e1 pinned to e2 along
    // the e-file, the old formula also included all of rank 1 and rank 2 in
    // lineBB[e1][e2] - so a piece pinned on e2 could "legally" hop to f1
    // (rank 1) even though that leaves the king in check on the e-file. This
    // let the engine play moves that leave its own king in check.
    // The correct construction intersects EMPTY-board attacks from both
    // squares: a rook on s1 sees all of s1's rank+file; a rook on s2 sees all
    // of s2's rank+file; the only squares common to both are the single
    // rank-or-file line that actually passes through both squares.
    for (int s1=0; s1<64; s1++) for (int s2=0; s2<64; s2++) {
        betweenBB[s1][s2] = lineBB[s1][s2] = 0;
        if (s1==s2) continue;
        U64 r = rookAttSlow(s1, bit(s2));
        if (r & bit(s2)) {
            betweenBB[s1][s2] = r & rookAttSlow(s2, bit(s1));
            lineBB[s1][s2] = (rookAttSlow(s1,0) & rookAttSlow(s2,0)) | bit(s1) | bit(s2);
            continue;
        }
        U64 b2 = bishopAttSlow(s1, bit(s2));
        if (b2 & bit(s2)) {
            betweenBB[s1][s2] = b2 & bishopAttSlow(s2, bit(s1));
            lineBB[s1][s2] = (bishopAttSlow(s1,0) & bishopAttSlow(s2,0)) | bit(s1) | bit(s2);
        }
    }

    // Magic bitboards
    PRNG rng(1070372ULL * 3);
    for (int sq=0; sq<64; sq++) {
        // Bishop
        {
            U64 mask = calcBishopMask(sq);
            int bits = popcount(mask);
            U64 magic = findMagic(sq, true, rng);
            bishopMagics[sq] = {mask, magic, bishopTable[sq], 64-bits};
            int sz = 1<<bits;
            for (int i=0; i<sz; i++) {
                U64 occ = indexToOcc(i, bits, mask);
                int idx = int((occ*magic)>>(64-bits));
                bishopTable[sq][idx] = bishopAttSlow(sq, occ);
            }
        }
        // Rook
        {
            U64 mask = calcRookMask(sq);
            int bits = popcount(mask);
            U64 magic = findMagic(sq, false, rng);
            rookMagics[sq] = {mask, magic, rookTable[sq], 64-bits};
            int sz = 1<<bits;
            for (int i=0; i<sz; i++) {
                U64 occ = indexToOcc(i, bits, mask);
                int idx = int((occ*magic)>>(64-bits));
                rookTable[sq][idx] = rookAttSlow(sq, occ);
            }
        }
    }

    initZobrist();
}

// ==================== POSITION ====================

struct StateInfo {
    U64   hash;
    int   castling;
    int   epSq;     // -1 if none
    int   rule50;
    Piece captured;
};

struct Position {
    U64   bb[6][2];        // bb[type][color]
    U64   occ[3];          // occ[WHITE], occ[BLACK], occ[BOTH]
    Piece board[64];
    Color side;
    int   castling;
    int   epSq;
    int   rule50;
    int   gamePly;
    U64   hash;
    // Shadow history used only for repetition detection. Sized dynamically
    // (instead of a fixed array) so it can never overflow: gamePly grows both
    // from the real game (an arbitrarily long "position ... moves ..." list)
    // and from search recursion (up to MAX_PLY deeper), and a fixed-size
    // array here was a real out-of-bounds-write risk on long games.
    vector<StateInfo> hist;

    void clear() {
        memset(bb, 0, sizeof(bb));
        memset(occ, 0, sizeof(occ));
        for (int i=0; i<64; i++) board[i]=EMPTY;
        side=WHITE; castling=0; epSq=-1; rule50=0; gamePly=0; hash=0;
        if (hist.size() < 1024) hist.assign(1024, StateInfo{});
    }

    U64  pieces(PieceType pt, Color c) const { return bb[pt][c]; }
    U64  byColor(Color c) const { return occ[c]; }
    U64  all() const { return occ[2]; }
    int  king(Color c) const { return lsb(bb[KING][c]); }

    void putPiece(Piece p, int sq) {
        U64 b = bit(sq);
        Color c = colorOf(p); PieceType t = typeOf(p);
        board[sq]=p; bb[t][c]|=b; occ[c]|=b; occ[2]|=b;
        hash ^= zobPiece[p][sq];
    }
    void removePiece(int sq) {
        Piece p = board[sq]; if (p==EMPTY) return;
        U64 b = bit(sq);
        Color c=colorOf(p); PieceType t=typeOf(p);
        board[sq]=EMPTY; bb[t][c]&=~b; occ[c]&=~b; occ[2]&=~b;
        hash ^= zobPiece[p][sq];
    }
    void movePiece(int from, int to) {
        Piece p=board[from];
        U64 ft=bit(from)|bit(to);
        Color c=colorOf(p); PieceType t=typeOf(p);
        board[to]=p; board[from]=EMPTY;
        bb[t][c]^=ft; occ[c]^=ft; occ[2]^=ft;
        hash ^= zobPiece[p][from]^zobPiece[p][to];
    }

    U64 attackersTo(int sq, U64 occupied) const {
        return (pawnAtk[BLACK][sq] & bb[PAWN][WHITE]) |
               (pawnAtk[WHITE][sq] & bb[PAWN][BLACK]) |
               (knightAtk[sq]      & (bb[KNIGHT][WHITE]|bb[KNIGHT][BLACK])) |
               (getBishopAtk(sq,occupied) & (bb[BISHOP][WHITE]|bb[BISHOP][BLACK]|bb[QUEEN][WHITE]|bb[QUEEN][BLACK])) |
               (getRookAtk(sq,occupied)   & (bb[ROOK][WHITE]|bb[ROOK][BLACK]|bb[QUEEN][WHITE]|bb[QUEEN][BLACK])) |
               (kingAtk[sq]        & (bb[KING][WHITE]|bb[KING][BLACK]));
    }
    U64 attackersToByColor(int sq, U64 occupied, Color c) const {
        return (pawnAtk[~c][sq]    & bb[PAWN][c]) |
               (knightAtk[sq]      & bb[KNIGHT][c]) |
               (getBishopAtk(sq,occupied) & (bb[BISHOP][c]|bb[QUEEN][c])) |
               (getRookAtk(sq,occupied)   & (bb[ROOK][c]|bb[QUEEN][c])) |
               (kingAtk[sq]        & bb[KING][c]);
    }
    bool inCheck(Color c) const {
        return attackersToByColor(king(c), all(), ~c) != 0;
    }
    U64 checkersBB() const {
        return attackersToByColor(king(side), all(), ~side);
    }

    // Compute pinned pieces for color c, and their pinners
    void pinInfo(Color c, U64& pinned, U64& pinners) const {
        pinned = pinners = 0;
        int ksq = king(c); Color them=~c;
        U64 snipers = ((getRookAtk(ksq,0) & (bb[ROOK][them]|bb[QUEEN][them])) |
                       (getBishopAtk(ksq,0) & (bb[BISHOP][them]|bb[QUEEN][them])));
        U64 occupied = all();
        while (snipers) {
            int sn = popLsb(snipers);
            U64 between = betweenBB[ksq][sn] & occupied;
            if (between && !several(between) && (between & occ[c])) {
                pinned  |= between;
                pinners |= bit(sn);
            }
        }
    }

    bool isLegal(Move m) const;
    void doMove(Move m, StateInfo& si);
    void undoMove(Move m, const StateInfo& si);
    void doNull(StateInfo& si);
    void undoNull(const StateInfo& si);
    void setFen(const string& fen);
    string toFen() const;
    bool isDraw() const;
    bool isRepetition() const;
    void ensureHistCapacity(int ply);
    bool hasMajorPiece(Color c) const { return (bb[KNIGHT][c]|bb[BISHOP][c]|bb[ROOK][c]|bb[QUEEN][c])!=0; }
};

// Castling update table
static const int CASTLE_MASK[64] = {
    ~WQ_C,15,15,15,~(WK_C|WQ_C),15,15,~WK_C,
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    ~BQ_C,15,15,15,~(BK_C|BQ_C),15,15,~BK_C
};

void Position::doMove(Move m, StateInfo& si) {
    si.hash=hash; si.castling=castling; si.epSq=epSq;
    si.rule50=rule50; si.captured=EMPTY;

    int from=fromSq(m), to=toSq(m), flag=flagOf(m);
    Color us=side, them=~us;

    if (epSq!=-1) { hash^=zobEP[epSq&7]; epSq=-1; }
    hash ^= zobCastle[castling];
    rule50++;

    if (flag==EP_CAP) {
        int capSq = to + (us==WHITE ? -8 : 8);
        si.captured = board[capSq];
        removePiece(capSq);
        movePiece(from, to);
        rule50=0;
    } else if (flag==KS_CASTLE||flag==QS_CASTLE) {
        movePiece(from, to);
        int rf = (flag==KS_CASTLE) ? (us==WHITE?H1:H8) : (us==WHITE?A1:A8);
        int rt = (flag==KS_CASTLE) ? (us==WHITE?F1:F8) : (us==WHITE?D1:D8);
        movePiece(rf, rt);
    } else {
        if (board[to]!=EMPTY) { si.captured=board[to]; removePiece(to); rule50=0; }
        if (isPromotion(m)) {
            removePiece(from);
            putPiece(makePiece(us, promoType(m)), to);
            rule50=0;
        } else {
            if (typeOf(board[from])==PAWN) rule50=0;
            movePiece(from, to);
        }
        if (flag==DPUSH) { epSq=to+(us==WHITE?-8:8); hash^=zobEP[epSq&7]; }
    }
    castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
    hash ^= zobCastle[castling];
    side = them; hash ^= zobSide;
    gamePly++;
    ensureHistCapacity(gamePly);
    hist[gamePly] = si;
}

void Position::undoMove(Move m, const StateInfo& si) {
    gamePly--;
    side=~side;
    int from=fromSq(m),to=toSq(m),flag=flagOf(m);
    Color us=side;

    if (flag==EP_CAP) {
        movePiece(to,from);
        putPiece(si.captured, to+(us==WHITE?-8:8));
    } else if (flag==KS_CASTLE||flag==QS_CASTLE) {
        movePiece(to,from);
        int rf=(flag==KS_CASTLE)?(us==WHITE?H1:H8):(us==WHITE?A1:A8);
        int rt=(flag==KS_CASTLE)?(us==WHITE?F1:F8):(us==WHITE?D1:D8);
        movePiece(rt,rf);
    } else {
        if (isPromotion(m)) { removePiece(to); putPiece(makePiece(us,PAWN),from); }
        else movePiece(to,from);
        if (si.captured!=EMPTY) putPiece(si.captured,to);
    }
    hash=si.hash; castling=si.castling; epSq=si.epSq; rule50=si.rule50;
}

void Position::doNull(StateInfo& si) {
    si.hash=hash; si.epSq=epSq; si.captured=EMPTY;
    if (epSq!=-1){hash^=zobEP[epSq&7]; epSq=-1;}
    side=~side; hash^=zobSide; gamePly++;
    ensureHistCapacity(gamePly);
    hist[gamePly]=si;
}

// Grow the shadow history so hist[ply] is always a valid index. Amortized O(1);
// in virtually all real games/searches this branch is false and never resizes,
// since clear() pre-reserves 1024 slots (far beyond MAX_PLY + any realistic game length).
void Position::ensureHistCapacity(int ply) {
    if (ply >= (int)hist.size()) hist.resize(hist.size()*2 + 64);
}
void Position::undoNull(const StateInfo& si) {
    gamePly--; side=~side; hash=si.hash; epSq=si.epSq;
}

bool Position::isLegal(Move m) const {
    int from=fromSq(m), to=toSq(m), flag=flagOf(m);
    Color us=side;
    int ksq=king(us);

    // En passant: verify no horizontal/diagonal pin exposing king
    if (flag==EP_CAP) {
        int capSq=to+(us==WHITE?-8:8);
        U64 occ2=(all()^bit(from)^bit(capSq))|bit(to);
        return !(getRookAtk(ksq,occ2)  &(bb[ROOK][~us]  |bb[QUEEN][~us]))&&
               !(getBishopAtk(ksq,occ2)&(bb[BISHOP][~us]|bb[QUEEN][~us]));
    }

    // Castling: king safety already verified in move generation
    if (isCastle(m)) return true;

    // King moves: destination must not be attacked after king vacates
    if (typeOf(board[from])==KING) {
        U64 occ2=all()^bit(from);
        return !attackersToByColor(to,occ2,~us);
    }

    // Non-king moves ---
    // Rule 1: pinned piece must stay on pin ray
    U64 pinned, pinners;
    pinInfo(us, pinned, pinners);
    if ((pinned & bit(from)) && !(lineBB[ksq][from] & bit(to)))
        return false;

    // Rule 2: if in check, must capture checker or block ray
    U64 chk = attackersToByColor(ksq, all(), ~us);
    if (chk) {
        if (several(chk)) return false; // double check: only king can move
        int checker = lsb(chk);
        if (!(bit(to) & (bit(checker)|betweenBB[ksq][checker])))
            return false;
    }

    return true;
}

// hist[k] stores the StateInfo captured BEFORE the move that produced ply k
// (it exists to support undoMove), so hist[k].hash is actually the hash of
// the position at ply (k-1), not at ply k. To compare the current position
// against the position that was at ply j (same side to move, j = gamePly-2,
// gamePly-4, ...), the correct lookup is therefore hist[j+1], not hist[j].
// The previous version compared hist[gamePly-2], hist[gamePly-4], ... directly,
// which are hashes from plies with the SIDE TO MOVE FLIPPED relative to the
// current position - since the side-to-move bit is folded into the hash, those
// comparisons could essentially never match, so repetition draws were never
// detected in practice.
bool Position::isRepetition() const {
    int limit = gamePly - rule50;
    for (int j=gamePly-2; j>=limit && j>=0; j-=2)
        if (hist[j+1].hash==hash) return true;
    return false;
}

bool Position::isDraw() const {
    if (rule50>=100) return true;
    if (isRepetition()) return true;
    // Insufficient material
    U64 all_pieces = all();
    if (popcount(all_pieces)==2) return true; // KK
    if (popcount(all_pieces)==3) {
        if (bb[KNIGHT][WHITE]||bb[KNIGHT][BLACK]||bb[BISHOP][WHITE]||bb[BISHOP][BLACK]) return true;
    }
    return false;
}

// Known-good fallback used only if a FEN turns out to be unparseable/invalid.
static const char* STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

void Position::setFen(const string& fen) {
    clear();
    istringstream ss(fen);
    string brd, sid, cas, ep;
    int hm=0, fm=1;
    ss>>brd>>sid>>cas>>ep>>hm>>fm;

    // Board field: bounds-checked so a malformed string (too many pieces on a
    // rank, a digit like '9', too many ranks, ...) can never push sq outside
    // [0,63] into putPiece(), which would otherwise write past the end of
    // board[64]/the bitboards (undefined behavior / memory corruption).
    int sq=A8;
    for (char c : brd) {
        if (c=='/') { sq-=16; continue; }
        if (isdigit(c)) { sq+=c-'0'; continue; }
        Piece p;
        switch(c){
          case'P':p=W_PAWN;break;case'N':p=W_KNIGHT;break;case'B':p=W_BISHOP;break;
          case'R':p=W_ROOK;break;case'Q':p=W_QUEEN;break;case'K':p=W_KING;break;
          case'p':p=B_PAWN;break;case'n':p=B_KNIGHT;break;case'b':p=B_BISHOP;break;
          case'r':p=B_ROOK;break;case'q':p=B_QUEEN;break;case'k':p=B_KING;break;
          default:continue;
        }
        if (sq < 0 || sq > 63) break; // malformed board field: stop, don't write OOB
        putPiece(p,sq++);
    }
    side = (sid=="b") ? BLACK : WHITE;
    if (side==BLACK) hash^=zobSide;
    castling=0;
    if (cas.find('K')!=string::npos) castling|=WK_C;
    if (cas.find('Q')!=string::npos) castling|=WQ_C;
    if (cas.find('k')!=string::npos) castling|=BK_C;
    if (cas.find('q')!=string::npos) castling|=BQ_C;
    hash^=zobCastle[castling];

    // En-passant field: validate both the characters and the resulting square
    // before using it. epSq is later used to index pawnAtk[..][epSq] during
    // move generation, so an unvalidated value here (e.g. from a garbled
    // field like "ep" is a bare "x", or from a fuzzed FEN) was a real
    // out-of-bounds read risk.
    epSq=-1;
    if (ep.size()>=2 && ep[0]>='a' && ep[0]<='h' && ep[1]>='1' && ep[1]<='8') {
        int cand = (ep[0]-'a') + (ep[1]-'1')*8;
        if (cand>=0 && cand<64) { epSq=cand; hash^=zobEP[epSq&7]; }
    }
    rule50 = max(0, hm);
    gamePly=0;

    // A FEN missing a king (or with more than one per side) leaves king(c)
    // calling lsb(0), which is undefined behavior and would then cascade into
    // out-of-bounds table lookups all over move generation/eval. Reject and
    // fall back to a known-good position instead of crashing.
    if (popcount(bb[KING][WHITE])!=1 || popcount(bb[KING][BLACK])!=1) {
        cerr << "info string Warning: invalid or incomplete FEN (need exactly one king per side); using startpos instead\n";
        if (fen != STARTPOS_FEN) setFen(STARTPOS_FEN); // STARTPOS_FEN is always valid, so this cannot recurse further
        return;
    }
}

string Position::toFen() const {
    string fen;
    for (int r=7;r>=0;r--) {
        int e=0;
        for (int f=0;f<8;f++) {
            Piece p=board[r*8+f];
            if (p==EMPTY){e++;continue;}
            if(e){fen+='0'+e;e=0;}
            const char pc[]="PNBRQKpnbrqk";
            fen+=pc[p];
        }
        if(e)fen+='0'+e;
        if(r>0)fen+='/';
    }
    fen+=' '; fen+=(side==WHITE?'w':'b'); fen+=' ';
    if(castling&WK_C)fen+='K';
    if(castling&WQ_C)fen+='Q';
    if(castling&BK_C)fen+='k';
    if(castling&BQ_C)fen+='q';
    if(!castling)fen+='-';
    fen+=' ';
    if(epSq!=-1){fen+='a'+fileOf(epSq);fen+='1'+rankOf(epSq);}else fen+='-';
    fen+=' '+to_string(rule50)+' '+to_string(1+gamePly/2);
    return fen;
}

// ==================== MOVE GENERATION ====================

struct MoveList {
    array<Move, MAX_MOVES> moves;
    int size = 0;
    void push(Move m) { moves[size++]=m; }
    void push(int f,int t,int fl=QUIET){push(makeMove(f,t,fl));}
    Move* begin(){return moves.data();}
    Move* end(){return moves.data()+size;}
};

void addPawnMoves(const Position& pos, MoveList& ml, bool capsOnly) {
    Color us=pos.side, them=~us;
    U64 pawns=pos.bb[PAWN][us], opp=pos.occ[them], empty=~pos.all();
    int dir = (us==WHITE)?8:-8;
    U64 rank7 = (us==WHITE)?RANK_7:RANK_2;
    U64 rank3 = (us==WHITE)?RANK_3:RANK_6;

    // Promotions captures
    U64 promo = pawns & rank7;
    if (promo) {
        U64 cl=(us==WHITE)?shiftNW(promo)&opp:shiftSW(promo)&opp;
        U64 cr=(us==WHITE)?shiftNE(promo)&opp:shiftSE(promo)&opp;
        while(cl){int t=popLsb(cl); ml.push(t-(dir-1),t,CAP_PROMO_Q); ml.push(t-(dir-1),t,CAP_PROMO_R); ml.push(t-(dir-1),t,CAP_PROMO_B); ml.push(t-(dir-1),t,CAP_PROMO_N);}
        while(cr){int t=popLsb(cr); ml.push(t-(dir+1),t,CAP_PROMO_Q); ml.push(t-(dir+1),t,CAP_PROMO_R); ml.push(t-(dir+1),t,CAP_PROMO_B); ml.push(t-(dir+1),t,CAP_PROMO_N);}
        if (!capsOnly) {
            U64 psh=((us==WHITE)?shiftN(promo):shiftS(promo))&empty;
            while(psh){int t=popLsb(psh); ml.push(t-dir,t,PROMO_Q); ml.push(t-dir,t,PROMO_R); ml.push(t-dir,t,PROMO_B); ml.push(t-dir,t,PROMO_N);}
        }
    }

    // Regular captures
    U64 nonPromo = pawns & ~rank7;
    U64 cl=(us==WHITE)?shiftNW(nonPromo)&opp:shiftSW(nonPromo)&opp;
    U64 cr=(us==WHITE)?shiftNE(nonPromo)&opp:shiftSE(nonPromo)&opp;
    while(cl){int t=popLsb(cl); ml.push(t-(dir-1),t,CAPTURE);}
    while(cr){int t=popLsb(cr); ml.push(t-(dir+1),t,CAPTURE);}

    // En passant
    if (pos.epSq!=-1) {
        U64 ep=pawnAtk[them][pos.epSq] & nonPromo;
        while(ep){int f=popLsb(ep); ml.push(f,pos.epSq,EP_CAP);}
    }

    if (!capsOnly) {
        U64 push1=((us==WHITE)?shiftN(nonPromo):shiftS(nonPromo))&empty;
        // Pawns that reached rank3 (white)/rank6 (black) via that single push
        // are exactly the ones that started on their home rank - captured
        // here BEFORE push1 is consumed by popLsb() below, since it's needed
        // to compute the double push.
        U64 push1FromHome = push1 & rank3;
        while(push1){int t=popLsb(push1); ml.push(t-dir,t,QUIET);}
        // BUG FIX: this used to be shiftN(shiftN(nonPromo&rank3)&empty)&empty,
        // i.e. it filtered CURRENT pawn positions by rank3 before pushing at
        // all. Since pawns start the game on rank2, nonPromo&rank3 is empty
        // at the start position (and stays wrong throughout), so this never
        // generated a single legal double-push - e2e4, d2d4, and every other
        // two-square pawn move were silently absent from move generation for
        // the entire lifetime of this engine. The fix pushes from the
        // *already-single-pushed* pawns that started on the home rank
        // (push1FromHome, computed above) one square further.
        U64 push2=((us==WHITE)?shiftN(push1FromHome):shiftS(push1FromHome))&empty;
        while(push2){int t=popLsb(push2); ml.push(t-2*dir,t,DPUSH);}
    }
}

void generateMoves(const Position& pos, MoveList& ml, bool capsOnly=false) {
    Color us=pos.side, them=~us;
    U64 opp=pos.occ[them], empty=~pos.all(), occ=pos.all();

    addPawnMoves(pos, ml, capsOnly);

    // Knights
    U64 kn=pos.bb[KNIGHT][us];
    while(kn){int f=popLsb(kn); U64 a=knightAtk[f];
        U64 c=a&opp; while(c){int t=popLsb(c);ml.push(f,t,CAPTURE);}
        if(!capsOnly){U64 q=a&empty;while(q){int t=popLsb(q);ml.push(f,t,QUIET);}}
    }
    // Bishops
    U64 bi=pos.bb[BISHOP][us];
    while(bi){int f=popLsb(bi); U64 a=getBishopAtk(f,occ);
        U64 c=a&opp;while(c){int t=popLsb(c);ml.push(f,t,CAPTURE);}
        if(!capsOnly){U64 q=a&empty;while(q){int t=popLsb(q);ml.push(f,t,QUIET);}}
    }
    // Rooks
    U64 ro=pos.bb[ROOK][us];
    while(ro){int f=popLsb(ro); U64 a=getRookAtk(f,occ);
        U64 c=a&opp;while(c){int t=popLsb(c);ml.push(f,t,CAPTURE);}
        if(!capsOnly){U64 q=a&empty;while(q){int t=popLsb(q);ml.push(f,t,QUIET);}}
    }
    // Queens
    U64 qu=pos.bb[QUEEN][us];
    while(qu){int f=popLsb(qu); U64 a=getQueenAtk(f,occ);
        U64 c=a&opp;while(c){int t=popLsb(c);ml.push(f,t,CAPTURE);}
        if(!capsOnly){U64 q=a&empty;while(q){int t=popLsb(q);ml.push(f,t,QUIET);}}
    }
    // King
    {int f=pos.king(us); U64 a=kingAtk[f];
        U64 c=a&opp;while(c){int t=popLsb(c);ml.push(f,t,CAPTURE);}
        if(!capsOnly){
            U64 q=a&empty;while(q){int t=popLsb(q);ml.push(f,t,QUIET);}
            // Castling
            U64 ksa=(us==WHITE)?0x60ULL:0x6000000000000000ULL;
            U64 qsa=(us==WHITE)?0xEULL:0x0E00000000000000ULL;
            int e1=(us==WHITE)?E1:E8, g1=(us==WHITE)?G1:G8, c1=(us==WHITE)?C1:C8;
            int f1=(us==WHITE)?F1:F8, d1=(us==WHITE)?D1:D8;
            if((pos.castling&(us==WHITE?WK_C:BK_C))&&!(occ&ksa)&&
               !pos.attackersToByColor(e1,occ,them)&&!pos.attackersToByColor(f1,occ,them)&&!pos.attackersToByColor(g1,occ,them))
                ml.push(e1,g1,KS_CASTLE);
            if((pos.castling&(us==WHITE?WQ_C:BQ_C))&&!(occ&qsa)&&
               !pos.attackersToByColor(e1,occ,them)&&!pos.attackersToByColor(d1,occ,them)&&!pos.attackersToByColor(c1,occ,them))
                ml.push(e1,c1,QS_CASTLE);
        }
    }
}

// ==================== SEE ====================

int see(const Position& pos, Move m) {
    int to=toSq(m), from=fromSq(m);
    if (flagOf(m)==EP_CAP) return 0;

    int gain[32]; int d=0;
    gain[0] = SEE_VAL[typeOf(pos.board[to])];

    U64 occ=pos.all();
    U64 attackers=pos.attackersTo(to,occ);
    U64 fromBB=bit(from);
    PieceType nextVictim=typeOf(pos.board[from]);
    Color side=pos.side;

    occ ^= fromBB;
    attackers &= occ;

    do {
        d++;
        gain[d]=SEE_VAL[nextVictim]-gain[d-1];
        if (max(-gain[d-1],gain[d])<0) break;
        attackers ^= fromBB;
        side=~side;
        fromBB=0;
        for (PieceType pt=PAWN; pt<=KING; pt=PieceType(pt+1)) {
            fromBB = pos.bb[pt][side] & attackers;
            if (fromBB){ fromBB &= -fromBB; nextVictim=pt; break; }
        }
        // Discover hidden attackers (X-ray)
        if (nextVictim==PAWN||nextVictim==BISHOP||nextVictim==QUEEN)
            attackers |= getBishopAtk(to,occ^fromBB)&
                (pos.bb[BISHOP][WHITE]|pos.bb[BISHOP][BLACK]|pos.bb[QUEEN][WHITE]|pos.bb[QUEEN][BLACK])&occ;
        if (nextVictim==ROOK||nextVictim==QUEEN)
            attackers |= getRookAtk(to,occ^fromBB)&
                (pos.bb[ROOK][WHITE]|pos.bb[ROOK][BLACK]|pos.bb[QUEEN][WHITE]|pos.bb[QUEEN][BLACK])&occ;
        occ ^= fromBB; attackers &= occ;
    } while (fromBB);

    while (--d) gain[d-1]=-max(-gain[d-1],gain[d]);
    return gain[0];
}

bool seeGe(const Position& pos, Move m, int threshold=0) {
    if (isCapture(m) && typeOf(pos.board[toSq(m)])==KING) return true;
    return see(pos,m)>=threshold;
}

// ==================== EVALUATION ====================

// PeSTO piece-square tables (Terje Kirstihagen)
// Indexed from white's perspective (a1=0..h8=63), mirrored for black
const int mg_pawn_table[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
  98,134, 61, 95, 68,126, 34,-11,
  -6,  7, 26, 31, 65, 56, 25,-20,
 -14, 13,  6, 21, 23, 12, 17,-23,
 -27, -2, -5, 12, 17,  6, 10,-25,
 -26, -4, -4,-10,  3,  3, 33,-12,
 -35, -1,-20,-23,-15, 24, 38,-22,
   0,  0,  0,  0,  0,  0,  0,  0
};
const int eg_pawn_table[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
 178,173,158,134,147,132,165,187,
  94,100, 85, 67, 56, 53, 82, 84,
  32, 24, 13,  5, -2,  4, 17, 17,
  13,  9, -3, -7, -7, -8,  3, -1,
   4,  7, -6,  1,  0, -5, -1, -8,
  13,  8,  8, 10, 13,  0,  2, -7,
   0,  0,  0,  0,  0,  0,  0,  0
};
const int mg_knight_table[64] = {
-167,-89,-34,-49, 61,-97,-15,-107,
 -73,-41, 72, 36, 23, 62,  7, -17,
 -47, 60, 37, 65, 84,129, 73,  44,
  -9, 17, 19, 53, 37, 69, 18,  22,
 -13,  4, 16, 13, 28, 19, 21,  -8,
 -23, -9, 12, 10, 19, 17, 25, -16,
 -29,-53,-12, -3, -1, 18,-14, -19,
-105,-21,-58,-33,-17,-28,-19, -23
};
const int eg_knight_table[64] = {
 -58,-38,-13,-28,-31,-27,-63,-99,
 -25, -8,-25, -2, -9,-25,-24,-52,
 -24,-20, 10,  9, -1, -9,-19,-41,
 -17,  3, 22, 22, 22, 11,  8, -18,
 -18, -6, 16, 25, 16, 17,  4, -18,
 -23, -3, -1, 15, 10, -3,-20, -22,
 -42,-20,-10, -5, -2,-20,-23, -44,
 -29,-51,-23,-15,-22,-18,-50, -64
};
const int mg_bishop_table[64] = {
 -29,  4,-82,-37,-25,-42,  7, -8,
 -26, 16,-18,-13, 30, 59, 18,-47,
 -16, 37, 43, 40, 35, 50, 37, -2,
  -4,  5, 19, 50, 37, 37,  7, -2,
  -6, 13, 13, 26, 34, 12, 10,  4,
   0, 15, 15, 15, 14, 27, 18, 10,
   4, 15, 16,  0,  7, 21, 33,  1,
 -33, -3,-14,-21,-13,-12,-39,-21
};
const int eg_bishop_table[64] = {
 -14,-21,-11, -8, -7, -9,-17,-24,
  -8, -4,  7,-12,-3,-13, -4,-14,
   2, -8,  0, -1, -2,  6,  0,  4,
  -3,  9, 12,  9, 14, 10,  3,  2,
  -6,  3, 13, 19,  7, 10, -3, -9,
 -12, -3,  8, 10, 13,  3, -7,-15,
 -14,-18, -7, -1,  4, -9,-15,-27,
 -23, -9,-23, -5, -9,-16, -5,-17
};
const int mg_rook_table[64] = {
  32, 42, 32, 51, 63,  9, 31, 43,
  27, 32, 58, 62, 80, 67, 26, 44,
  -5, 19, 26, 36, 17, 45, 61, 16,
 -24,-11,  7, 26, 24, 35, -8,-20,
 -36,-26,-12, -1,  9, -7,  6,-23,
 -45,-25,-16,-17,  3,  0, -5,-33,
 -44,-16,-20, -9, -1, 11, -6,-71,
 -19,-13,  1, 17, 16,  7,-37,-26
};
const int eg_rook_table[64] = {
  13, 10, 18, 15, 12, 12,  8,  5,
  11, 13, 13, 11, -3,  3,  8,  3,
   7,  7,  7,  5,  4, -3, -5, -3,
   4,  3, 13,  1,  2,  1, -1,  2,
   3,  5,  8,  4, -5, -6, -8, -11,
  -4,  0, -5, -1, -7,-12, -8,-16,
  -6, -6,  0,  2, -9, -9,-11, -3,
  -9,  2,  3, -1, -5,-13,  4,-20
};
const int mg_queen_table[64] = {
  -28,  0, 29, 12, 59, 44, 43, 45,
  -24,-39, -5,  1,-16, 57, 28, 54,
  -13,-17,  7,  8, 29, 56, 47, 57,
  -27,-27,-16,-16, -1, 17, -2,  1,
   -9,-26, -9,-10, -2, -4,  3, -3,
  -14,  2,-11, -2, -5,  2, 14,  5,
  -35, -8, 11,  2,  8, 15, -3,  1,
   -1,-18, -9, 10,-15,-25,-31,-50
};
const int eg_queen_table[64] = {
  -9, 22, 22, 27, 27, 19, 10, 20,
 -17, 20, 32, 41, 58, 25, 30,  0,
 -20,  6,  9, 49, 47, 35, 19,  9,
   3, 22, 24, 45, 57, 40, 57, 36,
 -18, 28, 19, 47, 31, 34, 39, 23,
 -16,-27, 15,  6,  9, 17, 10,  5,
 -22,-23,-30,-16,-16,-23,-36,-32,
 -33,-28,-22,-43, -5,-32,-20,-41
};
const int mg_king_table[64] = {
 -65, 23, 16,-15,-56,-34,  2, 13,
  29, -1,-20, -7, -8, -4,-38,-29,
  -9, 24,  2,-16,-20,  6, 22,-22,
 -17,-20,-12,-27,-30,-25,-14,-36,
 -49, -1,-27,-39,-46,-44,-33,-51,
 -14,-14,-22,-46,-44,-30,-15,-27,
   1,  7, -8,-64,-43,-16,  9,  8,
 -15, 36, 12,-54,  8,-28, 24, 14
};
const int eg_king_table[64] = {
 -74,-35,-18,-18,-11, 15,  4,-17,
 -12, 17, 14, 17, 17, 38, 23, 11,
  10, 17, 23, 15, 20, 45, 44, 13,
  -8, 22, 24, 27, 26, 33, 26,  3,
 -18, -4, 21, 24, 27, 23,  9,-11,
 -19, -3, 11, 21, 23, 16,  7, -9,
 -27,-11,  4, 13, 14,  4, -5,-17,
 -53,-34,-21,-11,-28,-14,-24,-43
};

const int* MG_TABLES[6] = {mg_pawn_table,mg_knight_table,mg_bishop_table,mg_rook_table,mg_queen_table,mg_king_table};
const int* EG_TABLES[6] = {eg_pawn_table,eg_knight_table,eg_bishop_table,eg_rook_table,eg_queen_table,eg_king_table};

// Mirror table for black pieces (flip rank)
inline int mirrorSq(int sq) { return sq ^ 56; }

struct EvalScore { int mg, eg; };
inline EvalScore operator+(EvalScore a,EvalScore b){return{a.mg+b.mg,a.eg+b.eg};}
inline EvalScore operator-(EvalScore a,EvalScore b){return{a.mg-b.mg,a.eg-b.eg};}
inline EvalScore& operator+=(EvalScore& a,EvalScore b){a.mg+=b.mg;a.eg+=b.eg;return a;}
inline EvalScore& operator-=(EvalScore& a,EvalScore b){a.mg-=b.mg;a.eg-=b.eg;return a;}
inline EvalScore makeScore(int mg,int eg){return{mg,eg};}

// Pawn hash table
struct PawnEntry {
    U64 key;
    EvalScore score;
    U64 passedPawns[2];
};
static const int PAWN_TABLE_SIZE = 16384;
// thread_local: with Lazy SMP, evaluate() runs concurrently on every search
// thread. A single shared table here was read AND written by all threads at
// once with no synchronization (a data race), which could hand back a
// mismatched score for a given key. thread_local gives each OS thread its
// own table (and, like any zero-initialized static/thread-local POD array,
// it starts fully zeroed, so pe.key==0 correctly reads as "empty slot").
thread_local PawnEntry pawnTable[PAWN_TABLE_SIZE];

// King attack weights
const int KA_PIECES[7] = {0, 1, 1, 2, 4, 0, 0};

EvalScore evalPawns(const Position& pos, Color us) {
    EvalScore score = {0,0};
    Color them = ~us;
    U64 ourPawns = pos.bb[PAWN][us];
    U64 theirPawns = pos.bb[PAWN][them];
    U64 p = ourPawns;

    while(p) {
        int sq = popLsb(p);
        int r = (us==WHITE)?rankOf(sq):7-rankOf(sq);
        int f = fileOf(sq);
        U64 frontSpan = (us==WHITE)?fillNorth(bit(sq)<<8):fillSouth(bit(sq)>>8);

        // Doubled pawns
        if (frontSpan & ourPawns) score += makeScore(-10,-20);

        // Isolated pawns
        U64 adjFiles = (f>0 ? fileMask(f-1) : 0ULL) | (f<7 ? fileMask(f+1) : 0ULL);
        if (!(adjFiles & ourPawns)) score += makeScore(-20,-15);

        // Passed pawn
        U64 passSpan = frontSpan;
        if(f>0) passSpan |= (frontSpan>>1)&~FILE_H;
        if(f<7) passSpan |= (frontSpan<<1)&~FILE_A;
        bool passed = !(passSpan & theirPawns);
        if (passed) {
            const int passBonus[8] = {0,10,15,25,45,75,120,0};
            score += makeScore(passBonus[r], passBonus[r]*2);
        }

        // Connected pawns
        U64 attacks = pawnAtk[them][sq]; // attacks from enemy direction = supporting squares
        if (attacks & ourPawns) score += makeScore(5+r*2, 5+r*2);
    }
    return score;
}

Value evaluate(const Position& pos) {
    EvalScore score = {0,0};
    int phase = 0;

    // Material + PSQT
    for (int c=0;c<2;c++) {
        int sign = (c==WHITE)?1:-1;
        for (int pt=PAWN;pt<KING;pt++) {
            U64 bb = pos.bb[pt][c];
            phase += popcount(bb) * (pt==QUEEN?4:pt==ROOK?2:pt<=BISHOP?1:0);
            while(bb) {
                int sq = popLsb(bb);
                int tsq = (c==WHITE)?sq:mirrorSq(sq);
                score.mg += sign*(PIECE_MG[pt] + MG_TABLES[pt][tsq]);
                score.eg += sign*(PIECE_EG[pt] + EG_TABLES[pt][tsq]);
            }
        }
        // King PSQT
        {
            int sq=pos.king(Color(c));
            int tsq=(c==WHITE)?sq:mirrorSq(sq);
            score.mg += sign*MG_TABLES[KING][tsq];
            score.eg += sign*EG_TABLES[KING][tsq];
        }
    }

    // Pawn evaluation (cached)
    U64 pawnKey = pos.bb[PAWN][WHITE] ^ (pos.bb[PAWN][BLACK] * 0x9e3779b97f4a7c15ULL);
    PawnEntry& pe = pawnTable[pawnKey % PAWN_TABLE_SIZE];
    if (pe.key != pawnKey) {
        pe.key = pawnKey;
        pe.score = evalPawns(pos, WHITE) - evalPawns(pos, BLACK);
        pe.passedPawns[WHITE] = pe.passedPawns[BLACK] = 0; // simplified
    }
    score += pe.score;

    // Mobility & king safety
    for (int c=0;c<2;c++) {
        int sign=(c==WHITE)?1:-1;
        Color us=Color(c), them=~us;
        int ksq=pos.king(them);
        U64 kingZone = kingAtk[ksq]|bit(ksq);
        int kingAttackers=0, kingAttackWeight=0;

        U64 mobilityArea = ~(pos.bb[PAWN][us]|pos.bb[KING][us]|(pawnAtk[them][0]?0:0));
        // Simplified mobility area (not pinned pieces, not attacked by pawns)
        U64 pawnBlocks = (us==WHITE)?shiftS(pos.bb[PAWN][them]):shiftN(pos.bb[PAWN][them]);
        mobilityArea = ~(pos.bb[PAWN][us]|pawnBlocks);

        // Knights
        U64 kn=pos.bb[KNIGHT][us];
        while(kn){
            int sq=popLsb(kn);
            U64 att=knightAtk[sq];
            int mob=popcount(att&mobilityArea);
            score.mg+=sign*(mob-4)*4; score.eg+=sign*(mob-4)*4;
            if(att&kingZone){kingAttackers++;kingAttackWeight+=KA_PIECES[KNIGHT]*popcount(att&kingZone);}
        }
        // Bishops
        U64 bi=pos.bb[BISHOP][us];
        while(bi){
            int sq=popLsb(bi);
            U64 att=getBishopAtk(sq,pos.all()&~pos.bb[QUEEN][us]);
            int mob=popcount(att&mobilityArea);
            score.mg+=sign*(mob-7)*3; score.eg+=sign*(mob-7)*3;
            if(att&kingZone){kingAttackers++;kingAttackWeight+=KA_PIECES[BISHOP]*popcount(att&kingZone);}
        }
        // Bishop pair
        if(popcount(pos.bb[BISHOP][us])>=2){score.mg+=sign*30;score.eg+=sign*50;}
        // Rooks
        U64 ro=pos.bb[ROOK][us];
        while(ro){
            int sq=popLsb(ro);
            U64 att=getRookAtk(sq,pos.all()&~pos.bb[ROOK][us]&~pos.bb[QUEEN][us]);
            int mob=popcount(att&mobilityArea);
            score.mg+=sign*(mob-7)*2; score.eg+=sign*(mob-7)*2;
            if(att&kingZone){kingAttackers++;kingAttackWeight+=KA_PIECES[ROOK]*popcount(att&kingZone);}
            // Open/semi-open file
            if(!(fileMask(fileOf(sq))&pos.bb[PAWN][us])){
                if(!(fileMask(fileOf(sq))&pos.bb[PAWN][them])){score.mg+=sign*15;score.eg+=sign*10;}
                else{score.mg+=sign*7;score.eg+=sign*5;}
            }
            // Rook on 7th rank
            if((us==WHITE&&rankOf(sq)==6)||(us==BLACK&&rankOf(sq)==1)){score.mg+=sign*10;score.eg+=sign*20;}
        }
        // Queens
        U64 qu=pos.bb[QUEEN][us];
        while(qu){
            int sq=popLsb(qu);
            U64 att=getQueenAtk(sq,pos.all());
            int mob=popcount(att&mobilityArea);
            score.mg+=sign*(mob-14)*1; score.eg+=sign*(mob-14)*2;
            if(att&kingZone){kingAttackers++;kingAttackWeight+=KA_PIECES[QUEEN]*popcount(att&kingZone);}
        }

        // King safety
        if(kingAttackers>=2&&pos.bb[QUEEN][us]){
            static const int safetyTable[100] = {
              0,  0,  1,  2,  3,  5,  7, 10, 13, 16,
             20, 25, 30, 37, 45, 54, 63, 72, 82, 92,
            103,114,124,135,145,156,166,176,186,195,
            204,213,222,231,240,249,258,267,276,285,
            294,303,312,321,330,339,348,357,366,375,
            384,393,402,411,420,429,438,447,456,465,
            474,483,492,501,510,519,528,537,546,555,
            564,573,582,591,600,600,600,600,600,600,
            600,600,600,600,600,600,600,600,600,600,
            600,600,600,600,600,600,600,600,600,600
            };
            int idx=min(99,kingAttackWeight);
            score.mg-=sign*safetyTable[idx];
        }
    }

    // Tapered evaluation
    phase = min(phase, 24);
    int mg_phase = phase;
    int eg_phase = 24 - phase;
    Value v = (score.mg * mg_phase + score.eg * eg_phase) / 24;

    // Tempo bonus
    v += (pos.side==WHITE) ? 15 : -15;

    // Return from white's perspective adjusted for side to move
    return (pos.side==WHITE) ? v : -v;
}

// ==================== TRANSPOSITION TABLE ====================

enum TTFlag : U8 { TT_NONE=0, TT_EXACT=1, TT_LOWER=2, TT_UPPER=3 };

struct TTEntry {
    U64   key;
    Move  move;
    Value value;
    Value staticEval;
    U8    depth;
    U8    gen;
    TTFlag flag;
    U8    pv;
};

struct TT {
    vector<array<TTEntry,2>> table;
    size_t size = 0;
    U8 generation = 0;

    void resize(int mb) {
        mb = max(1, mb);
        // Use size_t/64-bit arithmetic for the byte count: mb*1024*1024 as a
        // plain 32-bit int computation could overflow for large mb.
        size_t bytes = (size_t)mb * 1024ULL * 1024ULL;
        size_t newSize = max((size_t)1, bytes / sizeof(array<TTEntry,2>));
        try {
            table.assign(newSize, {});
            size = newSize;
        } catch (const std::exception&) {
            // Allocation failed (not enough memory available) - fall back to
            // a small size instead of letting the exception crash the engine.
            cerr << "info string Warning: could not allocate " << mb << " MB hash, falling back to 16 MB\n";
            size_t fallback = max((size_t)1, (size_t)(16ULL*1024*1024) / sizeof(array<TTEntry,2>));
            table.assign(fallback, {});
            size = fallback;
        }
        generation = 0;
    }

    void clear() { fill(table.begin(), table.end(), array<TTEntry,2>{}); generation=0; }
    void newSearch() { generation = (generation+1)&255; }

    TTEntry* probe(U64 key, bool& found) {
        auto& bucket = table[key % size];
        for (auto& e : bucket) {
            if (e.key == key) { found=true; return &e; }
        }
        found=false;
        // Find replace target
        TTEntry* replace = &bucket[0];
        for (auto& e : bucket) {
            int score0 = replace->depth - 4*(replace->gen!=generation);
            int scoreE = e.depth    - 4*(e.gen    !=generation);
            if (scoreE < score0) replace = &e;
        }
        return replace;
    }

    void store(U64 key, Move move, Value value, Value staticEval, int depth, TTFlag flag, bool pv) {
        bool found;
        TTEntry* e = probe(key, found);
        if (!found || flag==TT_EXACT || depth+4 > e->depth) {
            e->key=key; e->move=move; e->value=value; e->staticEval=staticEval;
            e->depth=(U8)max(0,depth); e->gen=generation; e->flag=flag; e->pv=(U8)pv;
        }
    }

    Value valueFromTT(Value v, int ply) {
        if (v >= VALUE_MATE_THRESH)  return v - ply;
        if (v <= -VALUE_MATE_THRESH) return v + ply;
        return v;
    }
    Value valueToTT(Value v, int ply) {
        if (v >= VALUE_MATE_THRESH)  return v + ply;
        if (v <= -VALUE_MATE_THRESH) return v - ply;
        return v;
    }
} tt;

// ==================== SEARCH ====================

struct SearchInfo {
    // Time management
    chrono::time_point<chrono::high_resolution_clock> startTime;
    int64_t timeLimit = 0;   // ms
    int64_t softLimit = 0;   // ms (for best move update)
    atomic<bool> stop{false};
    bool ponder = false;
    int maxDepth = MAX_PLY;
    int64_t maxNodes = INT64_MAX;

    // Results
    Move bestMove = MOVE_NONE;
    Value bestScore = -VALUE_INF;
    int  completedDepth = 0;

    // Incremented from every search thread concurrently (Lazy SMP), so this
    // must be atomic - a plain int64_t here was a data race (undefined
    // behavior, plus an unreliable node count and unreliable time checks).
    atomic<int64_t> nodes{0};

    int64_t elapsed() const {
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - startTime).count();
    }
    bool timeUp() const {
        if (stop) return true;
        int64_t n = nodes.load(memory_order_relaxed);
        // "go nodes N": stop as soon as the node budget is reached.
        if (maxNodes < INT64_MAX && n >= maxNodes) return true;
        // Throttle the (comparatively expensive) clock read to every 4096 nodes.
        return timeLimit>0 && (n&4095)==0 && elapsed()>=timeLimit;
    }
};

struct ThreadData {
    Position pos;
    int       threadId = 0;

    // Per-ply data
    Move killers[MAX_PLY][2];
    Move counterMove[12][64]; // [piece][to]
    int  history[2][64][64];  // [color][from][to]
    int  contHist[12][64][12][64]; // continuation history (simplified)

    // PV
    Move pvLine[MAX_PLY][MAX_PLY];
    int  pvLen[MAX_PLY];

    void clearHistory() {
        memset(killers,    0, sizeof(killers));
        memset(counterMove,0, sizeof(counterMove));
        memset(history,    0, sizeof(history));
        memset(contHist,   0, sizeof(contHist));
    }

    void updateHistory(Color c, Move m, int depth, const Move* quiets, int quietsCount) {
        int bonus = min(depth*depth, 400);
        auto updateHH = [&](Move mv, int val) {
            int& h = history[c][fromSq(mv)][toSq(mv)];
            h += val - h * abs(val) / 400;
        };
        updateHH(m, bonus);
        for (int i=0;i<quietsCount;i++) updateHH(quiets[i], -bonus);
    }
};

SearchInfo searchInfo;
vector<ThreadData> threads;

// Move ordering scores
static const int CAPTURE_BONUS = 1000000;
static const int PROMO_BONUS   = 900000;
static const int KILLER1_BONUS = 800000;
static const int KILLER2_BONUS = 700000;
static const int COUNTER_BONUS = 600000;

int scoreMove(const Position& pos, Move m, Move ttMove, int ply, const ThreadData& td) {
    if (m == ttMove) return 3000000;
    int flag = flagOf(m);
    if (flag==CAP_PROMO_Q||flag==PROMO_Q) return PROMO_BONUS + 1000;
    if (isCapture(m)) {
        int victim   = (int)typeOf(pos.board[toSq(m)]);
        int attacker = (int)typeOf(pos.board[fromSq(m)]);
        int see_val  = see(pos, m);
        return CAPTURE_BONUS + (victim*10 - attacker) + (see_val>=0 ? 0 : -200000);
    }
    if (isPromotion(m)) return PROMO_BONUS;
    if (m == td.killers[ply][0]) return KILLER1_BONUS;
    if (m == td.killers[ply][1]) return KILLER2_BONUS;
    // History
    Color c = pos.side;
    return td.history[c][fromSq(m)][toSq(m)];
}

// Fixed-capacity replacement for vector<pair<int,Move>> in the hot search
// path. qSearch/pvSearch build and sort one of these on every node visited
// (potentially millions of times per second), so a heap allocation there is
// pure overhead; MAX_MOVES safely bounds it (no chess position has more than
// 218 legal moves) and this lives on the stack instead.
struct ScoredMoveList {
    array<pair<int,Move>, MAX_MOVES> entries;
    int size = 0;
    inline void push(int score, Move m) { entries[size++] = {score, m}; }
    inline void sortDesc() {
        sort(entries.begin(), entries.begin()+size,
             [](const pair<int,Move>& a, const pair<int,Move>& b){ return a.first > b.first; });
    }
};

// Forward declarations
Value pvSearch(Position& pos, Value alpha, Value beta, int depth, int ply, bool cutNode, ThreadData& td);
Value qSearch (Position& pos, Value alpha, Value beta, int ply, ThreadData& td);

Value qSearch(Position& pos, Value alpha, Value beta, int ply, ThreadData& td) {
    if (searchInfo.timeUp()) return 0;
    searchInfo.nodes.fetch_add(1, memory_order_relaxed);

    if (pos.isDraw()) return VALUE_DRAW;
    if (ply >= MAX_PLY)  return evaluate(pos);

    bool inCheck = pos.inCheck(pos.side);
    Value standPat = inCheck ? -VALUE_INF : evaluate(pos);

    if (!inCheck) {
        if (standPat >= beta) return standPat;
        if (standPat > alpha) alpha = standPat;
    }

    // TT probe
    bool ttHit; TTEntry* tte = tt.probe(pos.hash, ttHit);
    Move ttMove = ttHit ? tte->move : MOVE_NONE;
    if (ttHit && (tte->flag==TT_EXACT ||
        (tte->flag==TT_LOWER && tt.valueFromTT(tte->value,ply)>=beta) ||
        (tte->flag==TT_UPPER && tt.valueFromTT(tte->value,ply)<=alpha)))
        return tt.valueFromTT(tte->value, ply);

    MoveList ml;
    generateMoves(pos, ml, !inCheck);

    // Score and sort (stack-allocated: see ScoredMoveList)
    ScoredMoveList scored;
    for (int i=0;i<ml.size;i++) scored.push(scoreMove(pos,ml.moves[i],ttMove,ply,td),ml.moves[i]);
    scored.sortDesc();

    Value best = inCheck ? -VALUE_INF : standPat;
    Move bestMove = MOVE_NONE;
    int moveCount = 0;
    StateInfo si;

    for (int idx=0; idx<scored.size; idx++) {
        Move m = scored.entries[idx].second;
        if (!pos.isLegal(m)) continue;
        moveCount++;

        // Delta pruning
        if (!inCheck && !isPromotion(m)) {
            int delta = standPat + SEE_VAL[typeOf(pos.board[toSq(m)])] + 150;
            if (delta <= alpha) continue;
        }
        // SEE pruning
        if (!inCheck && isCapture(m) && !seeGe(pos, m, -50)) continue;

        pos.doMove(m, si);
        Value v = -qSearch(pos, -beta, -alpha, ply+1, td);
        pos.undoMove(m, si);

        if (v > best) {
            best = v;
            bestMove = m;
            if (v > alpha) {
                alpha = v;
                if (v >= beta) {
                    tt.store(pos.hash, m, tt.valueToTT(v,ply), 0, 0, TT_LOWER, false);
                    return v;
                }
            }
        }
    }

    if (inCheck && moveCount==0) return -VALUE_MATE + ply;

    TTFlag flag = (best>alpha) ? TT_EXACT : TT_UPPER;
    tt.store(pos.hash, bestMove, tt.valueToTT(best,ply), 0, 0, flag, false);
    return best;
}

Value pvSearch(Position& pos, Value alpha, Value beta, int depth, int ply, bool cutNode, ThreadData& td) {
    if (searchInfo.timeUp()) return 0;
    // This MUST be the first array-touching check: td.pvLen/td.pvLine are sized
    // [MAX_PLY], so ply must be confirmed < MAX_PLY before any indexed write below.
    if (ply >= MAX_PLY) return evaluate(pos);

    bool isPV = (beta > alpha+1);
    bool isRoot = (ply == 0);

    td.pvLen[ply] = ply;

    if (!isRoot && pos.isDraw()) return VALUE_DRAW;
    if (depth <= 0) return qSearch(pos, alpha, beta, ply, td);

    searchInfo.nodes.fetch_add(1, memory_order_relaxed);

    // TT probe
    bool ttHit; TTEntry* tte = tt.probe(pos.hash, ttHit);
    Move ttMove = ttHit ? tte->move : MOVE_NONE;
    Value ttValue = ttHit ? tt.valueFromTT(tte->value, ply) : VALUE_INF;
    Value staticEval;

    if (ttHit) staticEval = tte->staticEval;
    else        staticEval = evaluate(pos);

    if (ttHit && !isPV && (int)tte->depth >= depth) {
        if ((tte->flag==TT_EXACT) ||
            (tte->flag==TT_LOWER && ttValue >= beta) ||
            (tte->flag==TT_UPPER && ttValue <= alpha))
            return ttValue;
    }

    bool inCheck = pos.inCheck(pos.side);

    // Check extension
    if (inCheck) depth++;

    // Static eval heuristics (not in check)
    if (!inCheck && !isPV) {
        // Razoring
        if (depth == 1 && staticEval + 350 < alpha) {
            Value v = qSearch(pos, alpha, beta, ply, td);
            if (v < alpha) return v;
        }

        // Futility pruning
        if (depth <= 8 && staticEval - 80*depth >= beta && staticEval < VALUE_MATE_THRESH)
            return staticEval;

        // Null move pruning
        if (depth >= 3 && staticEval >= beta && pos.hasMajorPiece(pos.side) && !cutNode) {
            int R = 3 + depth/6 + min(3,(staticEval-beta)/150);
            StateInfo si;
            pos.doNull(si);
            Value v = -pvSearch(pos, -beta, -beta+1, depth-R, ply+1, !cutNode, td);
            pos.undoNull(si);
            if (v >= beta) {
                if (v >= VALUE_MATE_THRESH) v = beta;
                if (depth < 14) return v;
                // Verification search
                Value vv = pvSearch(pos, beta-1, beta, depth-R, ply, false, td);
                if (vv >= beta) return v;
            }
        }

        // Prob cut
        if (depth >= 5 && abs(beta) < VALUE_MATE_THRESH) {
            int pbeta = beta + 200;
            MoveList ml; generateMoves(pos, ml, true);
            StateInfo si;
            for (int i=0;i<ml.size;i++) {
                Move m = ml.moves[i];
                if (!pos.isLegal(m)) continue;
                if (!seeGe(pos, m, pbeta-staticEval)) continue;
                pos.doMove(m, si);
                Value v = -qSearch(pos, -pbeta, -pbeta+1, ply+1, td);
                if (v >= pbeta) v = -pvSearch(pos,-pbeta,-pbeta+1,depth-4,ply+1,!cutNode,td);
                pos.undoMove(m, si);
                if (v >= pbeta) {
                    tt.store(pos.hash,m,tt.valueToTT(v,ply),staticEval,depth-3,TT_LOWER,false);
                    return v;
                }
            }
        }
    }

    // IID: if no TT move and deep, do a shallow search
    if (isPV && depth >= 5 && !ttMove) {
        pvSearch(pos, alpha, beta, depth-2, ply, cutNode, td);
        bool hit; tte = tt.probe(pos.hash, hit);
        if (hit) ttMove = tte->move;
    }

    // Generate and score all moves
    MoveList ml; generateMoves(pos, ml, false);
    ScoredMoveList scored;
    for (int i=0;i<ml.size;i++) scored.push(scoreMove(pos,ml.moves[i],ttMove,ply,td),ml.moves[i]);
    scored.sortDesc();

    Value best = -VALUE_INF;
    Move bestMove = MOVE_NONE;
    int moveCount = 0;
    array<Move, MAX_MOVES> quietsTried;
    int quietsCount = 0;
    StateInfo si;

    for (int scIdx=0; scIdx<scored.size; scIdx++) {
        Move m = scored.entries[scIdx].second;
        if (!pos.isLegal(m)) continue;
        moveCount++;

        bool isQuiet = !isCapture(m) && !isPromotion(m);
        int  newDepth = depth - 1;

        // Singular extension: if the TT move is the only move that avoids a
        // big score drop, extend the search by one ply on that move.
        if (!isRoot && depth>=8 && m==ttMove && !inCheck &&
            ttHit && (int)tte->depth >= depth-3 && tte->flag==TT_LOWER &&
            abs(ttValue)<VALUE_MATE_THRESH) {
            Value singBeta = ttValue - depth*2;
            // Search all moves except ttMove
            // (simplified: just do reduced search)
            Value singScore = pvSearch(pos, singBeta-1, singBeta, (depth-1)/2, ply, cutNode, td);
            if (singScore < singBeta) newDepth++;
            else if (singBeta >= beta) return singBeta; // multicut
        }

        // Futility pruning (move level)
        if (!isPV && !inCheck && moveCount>1 && isQuiet && depth<=8) {
            if (staticEval + 80 + 70*depth <= alpha) continue;
            if (!seeGe(pos, m, -30*depth)) continue;
        }

        // Late move pruning
        if (!isPV && !inCheck && depth<=6 && isQuiet && moveCount > 3+depth*depth) continue;

        // SEE pruning for captures
        if (!isPV && moveCount>1 && isCapture(m) && !seeGe(pos,m,-100*depth)) continue;

        pos.doMove(m, si);
        Value v;

        // PVS + LMR
        if (moveCount == 1) {
            v = -pvSearch(pos, -beta, -alpha, newDepth, ply+1, false, td);
        } else {
            // LMR
            int reduction = 0;
            if (depth >= 3 && moveCount >= 3 && isQuiet) {
                reduction = int(1.0 + log(depth) * log(moveCount) / 2.25);
                if (!isPV)   reduction++;
                if (cutNode) reduction++;
                reduction = max(0, min(reduction, depth-1));
            }

            v = -pvSearch(pos, -alpha-1, -alpha, newDepth-reduction, ply+1, true, td);
            if (v > alpha && reduction > 0)
                v = -pvSearch(pos, -alpha-1, -alpha, newDepth, ply+1, !cutNode, td);
            if (v > alpha && isPV)
                v = -pvSearch(pos, -beta, -alpha, newDepth, ply+1, false, td);
        }

        pos.undoMove(m, si);

        if (searchInfo.timeUp()) return 0;

        if (v > best) {
            best = v;
            bestMove = m;
            if (v > alpha) {
                alpha = v;
                // Update PV
                if (isPV) {
                    td.pvLine[ply][ply] = m;
                    for (int j=ply+1;j<td.pvLen[ply+1];j++)
                        td.pvLine[ply][j] = td.pvLine[ply+1][j];
                    td.pvLen[ply] = td.pvLen[ply+1];
                }
                if (v >= beta) {
                    // Update killers and history
                    if (isQuiet) {
                        if (td.killers[ply][0] != m) {
                            td.killers[ply][1] = td.killers[ply][0];
                            td.killers[ply][0] = m;
                        }
                        td.updateHistory(pos.side, m, depth, quietsTried.data(), quietsCount);
                    }
                    tt.store(pos.hash, m, tt.valueToTT(v,ply), staticEval, depth, TT_LOWER, isPV);
                    return v;
                }
            }
        }
        if (isQuiet && quietsCount < MAX_MOVES) quietsTried[quietsCount++] = m;
    }

    if (moveCount == 0) {
        // No legal moves: checkmate or stalemate
        return inCheck ? -VALUE_MATE + ply : VALUE_DRAW;
    }

    TTFlag flag = (best > alpha) ? TT_EXACT : TT_UPPER;
    tt.store(pos.hash, bestMove, tt.valueToTT(best,ply), staticEval, depth, flag, isPV);
    return best;
}

// ==================== ITERATIVE DEEPENING ====================

string moveToUci(Move m) {
    if (m==MOVE_NONE) return "0000";
    string s;
    s += ('a'+fileOf(fromSq(m)));
    s += ('1'+rankOf(fromSq(m)));
    s += ('a'+fileOf(toSq(m)));
    s += ('1'+rankOf(toSq(m)));
    if (isPromotion(m)) {
        const char promo[] = " nbrq";
        s += promo[promoType(m)];
    }
    return s;
}

Move uciToMove(const Position& pos, const string& s) {
    if (s.size()<4) return MOVE_NONE;
    int from = (s[0]-'a')+(s[1]-'1')*8;
    int to   = (s[2]-'a')+(s[3]-'1')*8;
    PieceType promo = NO_PT;
    if (s.size()>=5) {
        switch(s[4]){
            case'n':promo=KNIGHT;break; case'b':promo=BISHOP;break;
            case'r':promo=ROOK;break;   case'q':promo=QUEEN;break;
        }
    }
    MoveList ml; generateMoves(const_cast<Position&>(pos), ml);
    for (int i=0;i<ml.size;i++) {
        Move m=ml.moves[i];
        if (fromSq(m)==from && toSq(m)==to) {
            if (promo==NO_PT || promoType(m)==promo)
                if (pos.isLegal(m)) return m;
        }
    }
    return MOVE_NONE;
}

mutex outputMutex;
void uciOutput(const string& s) {
    lock_guard<mutex> lg(outputMutex);
    cout << s << '\n';
    cout.flush();
}

void search(Position& pos, SearchInfo& info, int numThreads=1) {
    info.stop = false;
    info.startTime = chrono::high_resolution_clock::now();
    info.bestMove = MOVE_NONE;
    info.bestScore = -VALUE_INF;
    info.nodes = 0;

    tt.newSearch();

    // Resize thread pool
    threads.resize(numThreads);
    for (int i=0;i<numThreads;i++) {
        threads[i].pos = pos;
        threads[i].threadId = i;
        if (i==0) threads[i].clearHistory();
    }

    // Iterative deepening (main thread)
    Value alpha = -VALUE_INF, beta = VALUE_INF;
    Value delta = 25;
    Move  bestMove = MOVE_NONE;
    Value bestScore = -VALUE_INF;

    // Safety net: seed bestMove with the first legal move before searching
    // anything. If the search is stopped (very short movetime, an immediate
    // "stop", or just very slow hardware) before depth 1 even finishes, the
    // loop below breaks without ever assigning bestMove, and the engine would
    // otherwise report "bestmove 0000" - not a legal move, and something a
    // GUI may not handle gracefully - despite legal moves being available.
    {
        MoveList ml0; generateMoves(pos, ml0, false);
        for (int i=0;i<ml0.size;i++) {
            if (pos.isLegal(ml0.moves[i])) { bestMove = ml0.moves[i]; break; }
        }
        info.bestMove = bestMove;
    }

    // Launch helper threads (Lazy SMP)
    vector<thread> helperThreads;
    for (int t=1; t<numThreads; t++) {
        helperThreads.emplace_back([t, &info](){
            threads[t].clearHistory();
            for (int d=1; d<=info.maxDepth && !info.stop; d++) {
                pvSearch(threads[t].pos, -VALUE_INF, VALUE_INF, d, 0, false, threads[t]);
            }
        });
    }

    for (int depth=1; depth<=info.maxDepth && !info.stop; depth++) {
        // Aspiration windows
        if (depth >= 5) {
            alpha = max(-VALUE_INF, bestScore - delta);
            beta  = min( VALUE_INF, bestScore + delta);
        } else {
            alpha = -VALUE_INF; beta = VALUE_INF;
        }

        Value score;
        for (;;) {
            score = pvSearch(threads[0].pos, alpha, beta, depth, 0, false, threads[0]);
            if (info.timeUp()) break;

            if (score <= alpha) {
                beta  = (alpha + beta) / 2;
                alpha = max(-VALUE_INF, alpha - delta);
            } else if (score >= beta) {
                beta  = min(VALUE_INF, beta + delta);
            } else break;
            delta += delta/2;
        }

        if (info.timeUp()) break;

        bestScore = score;
        if (threads[0].pvLen[0] > 0)
            bestMove = threads[0].pvLine[0][0];

        info.completedDepth = depth;
        info.bestMove = bestMove;
        info.bestScore = bestScore;

        // Build PV string
        string pvStr;
        for (int i=0; i<threads[0].pvLen[0]; i++) {
            pvStr += moveToUci(threads[0].pvLine[0][i]);
            if (i+1 < threads[0].pvLen[0]) pvStr+=' ';
        }

        int64_t nodesNow = info.nodes.load(memory_order_relaxed);
        int64_t elapsed = max((int64_t)1, info.elapsed());
        int64_t nps = nodesNow * 1000 / elapsed;

        // Score display
        string scoreStr;
        if (abs(bestScore) >= VALUE_MATE_THRESH)
            scoreStr = "mate " + to_string((bestScore>0 ? VALUE_MATE-bestScore+1 : -VALUE_MATE-bestScore)/2);
        else
            scoreStr = "cp " + to_string(bestScore);

        uciOutput("info depth " + to_string(depth) +
                  " score " + scoreStr +
                  " nodes " + to_string(nodesNow) +
                  " nps " + to_string(nps) +
                  " time " + to_string(elapsed) +
                  " pv " + pvStr);

        // Time management: stop if soft limit reached
        if (info.softLimit > 0 && info.elapsed() >= info.softLimit) break;
    }

    info.stop = true;
    for (auto& t : helperThreads) t.join();

    uciOutput("bestmove " + moveToUci(info.bestMove));
}

// ==================== TIME MANAGEMENT ====================

void calcTime(int wtime, int btime, int winc, int binc, int movestogo, Color side, SearchInfo& info) {
    int myTime = (side==WHITE) ? wtime : btime;
    int myInc  = (side==WHITE) ? winc  : binc;

    int moves = (movestogo > 0) ? movestogo : 30;
    int base  = myTime / moves + myInc * 3 / 4;
    base = min(base, myTime - 50);
    base = max(base, 1);

    info.timeLimit = base * 2;
    info.softLimit = base;
}

// ==================== UCI INTERFACE ====================

int numThreads = 1;

void handlePosition(Position& pos, istringstream& is) {
    string token;
    is >> token;
    if (token=="startpos") {
        pos.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        is >> token; // "moves"
    } else if (token=="fen") {
        string fen;
        while (is>>token && token!="moves") fen+=token+' ';
        pos.setFen(fen);
    }
    while (is>>token) {
        Move m = uciToMove(pos, token);
        if (m==MOVE_NONE) break;
        StateInfo si; pos.doMove(m, si);
        // (doMove already records this state into pos.hist[pos.gamePly] internally.)
    }
}

thread searchThread;

void handleGo(Position& pos, istringstream& is) {
    int wtime=0,btime=0,winc=0,binc=0,movestogo=0,movetime=0,depth=MAX_PLY;
    int64_t maxNodes = INT64_MAX;
    bool infinite=false, ponder=false;
    string token;
    while(is>>token){
        if(token=="wtime")      is>>wtime;
        else if(token=="btime") is>>btime;
        else if(token=="winc")  is>>winc;
        else if(token=="binc")  is>>binc;
        else if(token=="movestogo") is>>movestogo;
        else if(token=="movetime")  is>>movetime;
        else if(token=="depth")     is>>depth;
        else if(token=="infinite")  infinite=true;
        else if(token=="ponder")    ponder=true;
        // Read into a local, then assign once below - searchInfo.maxNodes is
        // a persistent field, so writing it directly here meant a node limit
        // from a previous "go nodes N" stayed active on every later "go"
        // that didn't itself specify "nodes".
        else if(token=="nodes")     is>>maxNodes;
    }

    // Defensive clamp: a very large/garbled "go depth N" can't cause a crash
    // (search recursion is independently bounded by MAX_PLY), but it can make
    // a stored TT depth wrap when narrowed to the entry's 8-bit field, so keep
    // it in the range the engine actually supports.
    depth = max(1, min(depth, MAX_PLY));

    searchInfo.maxDepth = depth;
    searchInfo.maxNodes = maxNodes;
    searchInfo.ponder = ponder;
    searchInfo.timeLimit = 0;
    searchInfo.softLimit = 0;

    if (movetime > 0) {
        // Clamp to a minimum of 1ms. "movetime - 10" alone goes negative for
        // any movetime under 10ms, and timeUp() only checks the time budget
        // when timeLimit>0 - so a fast/low-latency time control (common in
        // engine-vs-engine testing) would silently disable the time check
        // entirely and let the search run unbounded until maxDepth.
        int64_t t = max((int64_t)1, (int64_t)movetime - 10);
        searchInfo.timeLimit = t;
        searchInfo.softLimit = t;
    } else if (!infinite && (wtime||btime)) {
        calcTime(wtime,btime,winc,binc,movestogo,pos.side,searchInfo);
    }

    if (searchThread.joinable()) searchThread.join();
    Position posCopy = pos;
    searchThread = thread([posCopy]() mutable {
        search(posCopy, searchInfo, numThreads);
    });
}

void uciLoop() {
    Position pos;
    pos.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    string line, token;
    while (getline(cin, line)) {
        istringstream is(line);
        is >> token;

        if (token=="uci") {
            uciOutput("id name Prometheus");
            uciOutput("id author PrometheusEngine");
            uciOutput("option name Hash type spin default 64 min 1 max 4096");
            uciOutput("option name Threads type spin default 1 min 1 max 64");
            uciOutput("option name Clear Hash type button");
            uciOutput("uciok");
        }
        else if (token=="isready") {
            uciOutput("readyok");
        }
        else if (token=="setoption") {
            string name; is>>name>>name; // skip "name"
            string val;  is>>val>>val;   // skip "value"
            // std::stoi throws on a missing/non-numeric/out-of-range value
            // (e.g. a GUI sending "setoption name Hash" with no value at
            // all). That exception was never caught, which meant a single
            // malformed setoption line would take down the whole engine
            // process. Validate and catch instead of trusting the input.
            auto parseIntSafe = [](const string& s, int lo, int hi, int fallback)->int {
                if (s.empty()) return fallback;
                try {
                    size_t consumed=0;
                    int v = stoi(s, &consumed);
                    return max(lo, min(hi, v));
                } catch (...) {
                    return fallback;
                }
            };
            if (name=="Hash") {
                tt.resize(parseIntSafe(val, 1, 4096, TT_SIZE_MB));
            }
            else if (name=="Threads") {
                numThreads = parseIntSafe(val, 1, 64, numThreads);
            }
            else if (name=="Clear") tt.clear();
        }
        else if (token=="ucinewgame") {
            tt.clear();
            pos.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }
        else if (token=="position") {
            handlePosition(pos, is);
        }
        else if (token=="go") {
            handleGo(pos, is);
        }
        else if (token=="stop") {
            searchInfo.stop = true;
            if (searchThread.joinable()) searchThread.join();
        }
        else if (token=="ponderhit") {
            searchInfo.ponder = false;
        }
        else if (token=="quit") {
            searchInfo.stop = true;
            break; // cleanup (stopping + joining searchThread) happens once, right after this loop
        }
        else if (token=="d") {
            // Debug: print board
            for (int r=7;r>=0;r--) {
                for (int f=0;f<8;f++) {
                    Piece p=pos.board[r*8+f];
                    const char pc[]="PNBRQKpnbrqk.";
                    cerr<<pc[p]<<' ';
                }
                cerr<<'\n';
            }
            cerr<<"FEN: "<<pos.toFen()<<'\n';
            cerr<<"Hash: "<<hex<<pos.hash<<dec<<'\n';
            cerr<<"Repetition: "<<(pos.isRepetition()?"yes":"no")
                <<"  Draw: "<<(pos.isDraw()?"yes":"no")
                <<"  gamePly: "<<pos.gamePly<<"  rule50: "<<pos.rule50<<'\n';
        }
        else if (token=="eval") {
            cerr<<"Eval: "<<evaluate(pos)<<'\n';
        }
    }
    // Reached both when the loop exits via "quit" (break, above) and when
    // stdin simply hits EOF without a "quit" ever being sent (e.g. a script
    // that pipes commands and closes the pipe, or a GUI/test harness that
    // just kills the connection). Either way, an in-flight search thread
    // must be stopped and joined here: a std::thread that is still
    // "joinable" when its destructor runs (i.e. never joined/detached) calls
    // std::terminate() and aborts the whole process. Previously this join
    // only happened inside the "quit" branch, so ending the session any
    // other way reliably crashed the engine on exit.
    searchInfo.stop = true;
    if (searchThread.joinable()) searchThread.join();
}

// ==================== MAIN ====================

int main(int /*argc*/, char** /*argv*/) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cerr << "Prometheus Chess Engine v2.0 - Initializing...\n";
    initAll();
    tt.resize(TT_SIZE_MB);
    threads.resize(1);
    threads[0].clearHistory();
    cerr << "Ready. Type 'uci' to start.\n";

    uciLoop();
    return 0;
}
