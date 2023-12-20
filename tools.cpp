// #pragma GCC optimize("O3")
// #include <atcoder/all>
// #include <bits/stdc++.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <bitset>
#include <algorithm>
#include <string>
#include <queue>
#include <stack>
#include <list>
#include <numeric>
#include <cassert> // assert();
#include <iomanip> // cout << setprecision(15); cout << setfill('0') << std::right << setw(3);
#include <optional> // optional<int> f = nullopt; if(f) f.value();
#include <regex> // regex_replace("target", regex("old"), "new");
#include <random>
#define MY_PI     3.14159265358979323846
#define MY_E      2.7182818284590452354
#define INF     (INT_MAX / 2)
#define LINF    (LLONG_MAX / 2)
#define FOR(i, a, b) for(int i = (a); i < (b); ++i)
#define RFOR(i, a, b) for(int i = (b) - 1; i >= (a); --i)
#define REP(i, n) FOR(i, 0, n)
#define RREP(i, n) RFOR(i, 0, n)
#define EACH(e, v) for(auto &e : v)
#define ITR(it, v) for(auto it = (v).begin(); it != (v).end(); ++it)
#define RITR(it, v) for(auto it = (v).rbegin(); it != (v).rend(); ++it)
#define CASE break; case
#define ALL(v)  (v).begin(), (v).end()
#define RALL(v) (v).rbegin(), (v).rend()
#define SZ(v) int((v).size())
#define EXIST(s, e) ((s).find(e) != (s).end())
#define SORT(v) sort((v).begin(), (v).end())
#define RSORT(v) sort((v).rbegin(), (v).rend())
#define SUM(v, type) accumulate((v).begin(), (v).end(), (type) 0)
#define MIN(v) (*min_element((v).begin(), (v).end()))
#define MAX(v) (*max_element((v).begin(), (v).end()))
#define ARGMIN(v) (min_element((v).begin(), (v).end()) - (v).begin())
#define ARGMAX(v) (max_element((v).begin(), (v).end()) - (v).begin())
#define REVERSE(v) reverse((v).begin(), (v).end())
#define ARANGE(v) iota((v).begin(), (v).end(), 0)
#define FILTER(src, tgt, func) copy_if(begin(src), end(src), back_inserter(tgt), func); // func = [](type x){return 条件;}
#define CTOI(c) (c - '0')
#define HEADSTR(str, n) str.substr(0, (n))
#define TAILSTR(str, n) str.substr((str).length() - (n))
#define CONTAINS(str, c) ((str).find(c) != string::npos)
#define INSPOS(v, a) (lower_bound((v).begin(), (v).end(), a) - (v).begin())
// デバッグ用
#define dump(x)  cerr << #x << " = " << (x) << endl;
#define debug(x) cerr << #x << " = " << (x) << " (L" << __LINE__ << ")" << " " << __FILE__ << endl;

using namespace std;

template<class T> vector<size_t> argsort(const vector<T> &vec, bool asc=true){
    vector<size_t> index(vec.size()); iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&vec, &asc](size_t i, size_t j){return asc ? (vec[i] < vec[j]):(vec[i] > vec[j]);});
    return index;
}
// 表示系
template<class T1, class T2> ostream& operator<<(ostream& os, const pair<T1, T2>& p) {
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}
template<class... T> ostream& operator<<(ostream& os, const tuple<T...>& t) {
    os << "("; apply([&os](auto&&... args) {((os << args << ", "), ...);}, t);
    os << ")"; return os;
}
template<class T> ostream& operator<<(ostream& os, const vector<T>& vec) {
    os << "[ "; for ( const T& item : vec ) os << item << ", ";
    os << "]"; return os;
}
template<class T> ostream& operator<<(ostream& os, const set<T>& s) {
    os << "{ "; for ( const T& item : s ) os << item << ", ";
    os << "}"; return os;
}
template<class T> ostream& operator<<(ostream& os, const multiset<T>& s) {
    os << "{ "; for ( const T& item : s ) os << item << ", ";
    os << "}"; return os;
}
template<class T1, class T2> ostream& operator<<(ostream& os, const map<T1, T2>& m) {
    os << "{ "; for ( const auto &[key, value] : m ) os << key << ":"<< value << ", ";
    os << "}"; return os;
}
template <class Head> void OUT(Head&& head) {cout << head << endl;}
template <class Head, class... Tail> void OUT(Head&& head, Tail&&... tail) {cout << head << " ";OUT(forward<Tail>(tail)...);}
// 入力系
template<class T1, class T2> istream& operator>>(istream& is, pair<T1, T2>& p) {
    is >> p.first >> p.second;
    return is;
}
template<class... T> istream& operator>>(istream& is, tuple<T...>& t) {
    apply([&is](auto&&... args) {((is >> args), ...);}, t);
    return is;
}
template<class T> istream& operator>>(istream& is, vector<T>& vec) {
    for ( T& item : vec ) is >> item;
    return is;
}
// 集合演算
template<class T> set<T> operator&(const set<T>& a, const set<T>& b) {// 共通集合
    set<T> ans; set_intersection(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}
template<class T> set<T> operator|(const set<T>& a, const set<T>& b) {// 和集合
    set<T> ans; set_union(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}
template<class T> set<T> operator-(const set<T>& a, const set<T>& b) {// 差集合
    set<T> ans; set_difference(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}

using LL = long long; using ULL = unsigned long long;
using VI = vector<int>; using VVI = vector<VI>; using VVVI = vector<VVI>;
using VL = vector<LL>; using VVL = vector<VL>; using VVVL = vector<VVL>;
using VB = vector<bool>; using VVB = vector<VB>; using VVVB = vector<VVB>;
using VD = vector<double>; using VVD = vector<VD>; using VVVD = vector<VVD>;
using VC = vector<char>; using VS = vector<string>;
using PII = pair<int,int>; using PLL = pair<LL,LL>; using PDD = pair<double,double>;
using MII = map<int,int>; using MLL = map<LL,LL>;
using SI = set<int>; using SL = set<LL>;
using MSI = multiset<int>; using MSL = multiset<LL>;
template<class T> using MAXPQ = priority_queue<T>;
template<class T> using MINPQ = priority_queue< T, vector<T>, greater<T> >;

istringstream debug_iss(R"(
デバッグ時はここに入力を貼り付けて下記マクロのコメントアウトを外す
)");
// #define cin debug_iss

template <class Head> void IN(Head&& head) {cin >> head;}
template <class Head, class... Tail> void IN(Head&& head, Tail&&... tail) {cin >> head;IN(forward<Tail>(tail)...);}

// --------------------ここから--------------------

vector<string> split_str(const string &str, char delim, bool ignore_empty){
    stringstream ss{str};
    string buf;
    vector<string> result;
    while (std::getline(ss, buf, delim)) {
        if (!ignore_empty || !buf.empty())
            result.emplace_back(buf);
    }
    if (!ignore_empty && !str.empty() && str.back()==delim)
        result.emplace_back("");
    return result;
}

template<class T>
vector<T> cumsum(const vector<T> &vec, bool zero_start){
    int n = SZ(vec) + zero_start;
    vector<T> cumsum(n);
    cumsum[0] = zero_start ? 0 : vec[0];
    FOR(i, 1, n)
        cumsum[i] = cumsum[i-1] + vec[i - zero_start];
    return cumsum;
}

// a^n % mod
template<class T>
T pow_mod(T a, T n, T mod){
    T ans = 1;
    a %= mod;
    while (n > 0) {
        if ((n & 1))
            ans = (ans * a) % mod;
        a = (a * a) % mod;
        n >>= 1;
    }
    return ans;
}

// 拡張ユークリッド互除法 ax + by = gcd(a,b) -> {x, y, gcd}
template<class T>
tuple<T,T,T> extend_euclid(T a, T b) {
    //STEP1
    T x0 = 1, x1 = 0, x2, y0 = 0, y1 = 1, y2, r0 = a, r1 = b, r2, q2;
    //STEP2
    while (r1 != 0) {
        //STEP2-1
        q2 = r0 / r1;
        r2 = r0 % r1;
        //STEP2-2
        x2 = x0 - q2 * x1;
        y2 = y0 - q2 * y1;
        //
        x0 = x1;
        x1 = x2;
        y0 = y1;
        y1 = y2;
        r0 = r1;
        r1 = r2;
    }
    return {x0, y0, r0};
}

// aのmod pにおける逆元 (aとpは互いに素)
template<class T>
T inverse(T a, T prime_mod, bool euclid=true){
    if (!euclid) 
        return pow_mod(a, prime_mod-2, prime_mod);
    auto [x,y,g]=extend_euclid(a, prime_mod);
    assert(g==1);
    return ( (x<0) ? (x+prime_mod):x ) % prime_mod;
}

// nCr
template<class T>
T combi(T n, T r){
    r = (r > n / 2) ? n - r : r;
    T ans = 1;
    for (T i = 1; i <= r; i++, n--) {
        ans *= n;
        ans /= i;
    }
    return ans;
}

// nCr % mod ただし modは素数
template<class T>
T combi_mod(T n, T r, T mod){
    T ans=1, div=1;
    for (T i = n; i > n-r; --i)
        ans = (ans*(i%mod))%mod;
    for (T i = 1; i < r+1; ++i)
        div = (div*(i%mod))%mod;
    T inv_div = inverse(div, mod); // 整数の逆元
    return (ans * inv_div) % mod;
}

// ceil(sqrt(n))を返す
template<class T>
T sqrt_int(T n){
    T left=0, right=n;
    while(left <= right){
        T mid = (left+right)/2;
        T val = mid*mid;
        if (val < n) left=mid+1;
        else right=mid-1;
    }
    return left;
}

// 素因数分解
template<class T>
vector<T> pfact(T n){
    vector<T> ans;
    for (T i=2; i*i<=n; ++i){
        while(n%i==0){
            n/=i;
            ans.emplace_back(i);
        }
    }
    if (n!=1)
        ans.emplace_back(n);
    return ans;
}

// 約数列挙
template<class T>
vector<T> divisor(T n){
    vector<T> ans;
    for (T i=1; i*i<=n; ++i){
        if(n%i!=0) continue;
        ans.emplace_back(i);
        if (i!=(n/i)) ans.emplace_back(n/i);
    }
    SORT(ans);
    return ans;
}

// エラトステネスのふるい: n以下の素数列挙
template<class T>
vector<T> eratosthenes(T n){
    vector<T> ans;
    VB is_prime(n+1, true);
    for (T i=2; i*i<=n; ++i){
        if(!is_prime[i]) continue;
        for(T j=2*i; j<=n; j+=i)
            is_prime[j] = false;
    }
    for (T i=2; i<=n; ++i)
        if (is_prime[i]) ans.emplace_back(i);
    return ans;
}

// 10進数からb進数に変換
template<class T> 
string to_baseB(T x, int b){
    string ans;
    do{
        int num = x%b;
        ans = (char)( (num<=9) ? ('0'+num) : ('A'+num-10) ) + ans;
        x/=b;
    }while(x!=0);
    return ans;
}

// b進数から10進数に変換
template<class T>
T to_base10(const string &x, int b){
    T ans=0, base=1;
    for(int i=x.length()-1; i>=0; --i){
        int num = ('0'<=x[i]&&x[i]<='9') ? (x[i]-'0') : (x[i]-'A'+10);
        ans+=base*num;
        base*=b;
    }
    return ans;
}

// 分数
template<class T>
class Frac {
    //a,bともに整数系のクラス
    T a, b; //  a/b
    void trim(){
        if (a == 0) {
            b = 1;
            return;
        }
        assert(b != 0);
        T g = gcd(abs(a), abs(b));
        a /= g, b /= g;
    }
public:
    static const Frac<T> ZERO;
    static const Frac<T> ONE;

    Frac(){ a = 0, b = 1; }
    Frac(const T& a, const T& b = 1){ this->a = a, this->b = b; trim(); }

    T getBelow()const{ return abs(b); }
    T getAbove()const{ return abs(a); }
    int getSign()const{
        if (a == 0) return 0;
        if ((a > 0) ^ (b > 0)) return -1;
        return 1;
    }
    void setValue(const T& a, const T& b = 1){ this->a = a, this->b = b; trim(); }

    Frac<T> add(const Frac<T> &f)const{
        Frac<T> ans;
        T g = gcd(abs(b), abs(f.b));
        ans.a = (f.b / g) * a + (b / g) * f.a;
        ans.b = (b / g) * f.b;
        ans.trim();
        return ans;
    }
    Frac<T> subtract(const Frac<T> &f)const {
        Frac<T> ans = f;
        ans.a = -ans.a;
        return this->add(ans);
    }
    Frac<T> multiply(const Frac<T> &f)const {
        Frac<T> ans;
        ans.a = a * f.a;
        ans.b = b * f.b;
        ans.trim();
        return ans;
    }
    Frac<T> divide(const Frac<T> &f) const {
        Frac<T> ans;
        ans.a = a * f.b;
        ans.b = b * f.a;
        ans.trim();
        return ans;
    }
    Frac<T> inverse()const{ return Frac<T>(b, a); }
    Frac<T> pow(const int& n)const{
        Frac<T> ans = ONE, tmp = *this;
        int exp = n;
        if (n < 0) {
            exp = -n;
            tmp = inverse();
        }
        if (exp & 1)ans = tmp;
        while (exp != 0) {
            tmp = tmp.multiply(tmp);
            if ((exp >>= 1)&1)ans = ans.multiply(tmp);
        }
        return ans;
    }
    bool equal(const Frac<T> &f)const{
        if (getSign() == f.getSign() && abs(a) == abs(f.a) && abs(b) == abs(f.b)) 
            return true;
        return false;
    }
    void print(std::ostream &os = std::cout)const{
        if (getSign() < 0)os << '-';
        os << abs(a);
        if (abs(b) != 1) os << '/' << abs(b);
    }

    bool isInt()const { return (abs(b) == 1); }
    T toInt() const { return a / b; }
    double toDouble() const { return (double) a / b; }
    static Frac<T> createOne(){ Frac<T> one; one.a = 1, one.b = 1; return one; }

    Frac<T>& operator=(const T& n) { setValue(n); return *this; }
    Frac<T> operator++() { *this = *this+ONE; return *this; }
    Frac<T> operator++(int n) { Frac<T> tmp = *this; *this = *this+ONE; return tmp; }
    Frac<T> operator--() { *this = *this-ONE; return *this; }
    Frac<T> operator--(int n) { Frac<T> tmp = *this; *this = *this-ONE; return tmp; }
    Frac<T> operator-() const { return Frac<T>(-a, b); }
    Frac<T> operator+() const { return *this; }
    Frac<T> operator+(const Frac<T> &f) const { return this->add(f); }
    Frac<T> operator-(const Frac<T> &f) const { return this->subtract(f); }
    Frac<T> operator*(const Frac<T> &f) const { return this->multiply(f); }
    Frac<T> operator/(const Frac<T> &f) const { return this->divide(f); }
    Frac<T> operator+(const T& n) const { return this->add(Frac<T>(n, 1)); }
    Frac<T> operator-(const T &n) const { return this->subtract(Frac<T>(n, 1)); }
    Frac<T> operator*(const T& n) const { return this->multiply(Frac<T>(n, T(1)));}
    Frac<T> operator/(const T& n) const { return this->divide(Frac<T>(n, T(1))); }
    Frac<T> operator ^(const int& n) const { return pow(n); }
    Frac<T>& operator+=(const Frac<T> &f) { *this = *this+f; return *this; }
    Frac<T>& operator-=(const Frac<T> &f) { *this = *this-f; return *this; }
    Frac<T>& operator*=(const Frac<T> &f) { *this = *this*f; return *this; }
    Frac<T>& operator/=(const Frac<T> &f) { *this = *this / f; return *this; }
    Frac<T>& operator+=(const T& n) { *this = *this+n; return *this; }
    Frac<T>& operator-=(const T& n) { *this = *this-n; return *this; }
    Frac<T>& operator*=(const T& n) { *this = *this*n; return *this; }
    Frac<T>& operator/=(const T& n) { *this = *this / n; return *this; }
    bool operator==(const Frac<T>& f) const { return equal(f); }
    bool operator<(const Frac<T> &f)const { return subtract(f).getSign() < 0; }
    bool operator>(const Frac<T> &f)const { return subtract(f).getSign() > 0; }
    bool operator<=(const Frac<T> &f)const { return subtract(f).getSign() <= 0; }
    bool operator>=(const Frac<T> &f)const { return subtract(f).getSign() >= 0; }
    bool operator!=(const Frac<T> &f)const { return !equal(f); }
    bool operator==(const T &n)const { return equal(Frac<T>(n)); }
    bool operator<(const T &n)const { return subtract(Frac<T>(n)).getSign() < 0; }
    bool operator>(const T &n)const { return subtract(Frac<T>(n)).getSign() > 0; }
    bool operator<=(const T &n)const { return subtract(Frac<T>(n)).getSign() <= 0; }
    bool operator>=(const T &n)const { return subtract(Frac<T>(n)).getSign() >= 0; }
    bool operator!=(const T &n)const { return !equal(Frac<T>(n)); }

    template<class U> friend Frac<U> operator+(const U&n, const Frac<U> &f) { return f + n; }
    template<class U> friend Frac<U> operator-(const U& n, const Frac<U> &f) { return -f + n; }
    template<class U> friend Frac<U> operator*(const U& n, const Frac<U> &f) { return f*n; }
    template<class U> friend Frac<U> operator/(const U& n, const Frac<U> &f) { return n * f.inverse(); }
    template<class U> friend std::ostream& operator<<(std::ostream& os, const Frac<U>& f) { f.print(os); return os; }
};
template<class T> const Frac<T> Frac<T>::ONE = createOne();
template<class T> const Frac<T> Frac<T>::ZERO;


template<class T>
optional<T> __simplex(vector<vector<T>> &a, int n_row, int n_col, VI& index){
    while(true){
        // 列選択
        int y=-1;
        for (int k = 1; k < n_col; k++)if(a[n_row - 1][k] < 0){
            y=k;
            break;
        }
        if (y < 0) break;

        // 行選択
        int x=-1, base_index=INF;
        T minval;
        for (int k = 0; k < n_row - 1; k++)if (a[k][y] > 0) {
            T p = a[k][0] / a[k][y];
            if (x<0 || p < minval) {
                minval=p, x = k, base_index=index[k];
            }else if(p==minval && index[k]<base_index){
                x = k, base_index=index[k];
            }
        }
        // 解なし
        if(x<0)
            return nullopt;
        // ピボット係数
        T p = a[x][y];

        // ピボット係数を p で除算
        for (int k = 0; k < n_col; k++)
            a[x][k] = a[x][k] / p;

        // ピボット列の掃き出し
        for (int k = 0; k < n_row; k++)if (k != x) {
            T d = a[k][y];
            for (int j = 0; j < n_col; j++)
                a[k][j] = a[k][j] - d * a[x][j];
        }
        // 基底変数の添字変更
        index[x]=y;
    }
    return a[n_row - 1][0];
}

// max  cx
// s.t. Ax<=b, x>=0
// is_eq: 等号かどうか
// NOTE: double型で使うには誤差を許容させる必要がある
template<class T>
optional<T> simplex(const vector<vector<T>> &A, const vector<T> &b, const VB& is_eq, const vector<T> &c){
    // 人為変数, スラック変数
    int n_art=0, n_slk=0;
    REP(i, SZ(b)){
        if(!is_eq[i])n_slk++;
        if(is_eq[i] || b[i]<0)
            n_art++;
    }
    // 変数
    int n_var=SZ(c);
    // 行数
    int n_row = SZ(A)+1;
    // 列数
    int n_col = 1+n_var+n_slk+n_art;
    // 基底
    VI index(SZ(A));
    // 単体表: [定数, 変数, スラック, 人為]
    vector<vector<T>> a(n_row, vector<T>(n_col, 0));
    int na=0, ns=0;
    REP(i, SZ(A)){
        // 定数列
        a[i][0]=b[i]<0 ? -b[i] : b[i];
        // 変数列
        REP(j, n_var){
            a[i][1+j]= b[i]<0 ?-A[i][j]: A[i][j];
        }
        // スラック変数
        if(!is_eq[i])
            a[i][1+n_var+ns]=b[i]<0 ? -1 : 1;
        // 基底
        if(is_eq[i] || b[i]<0)
            index[i]=1+n_var+n_slk+na;
        else
            index[i]=1+n_var+ns;
        // 位置
        if(is_eq[i] || b[i]<0)
            na++;
        if(!is_eq[i])
            ns++;
    }
    if(n_art!=0){
        // 1st stageの目的関数
        // 目的関数(sum -人為変数)に含まれる基底を除外する
        // 人為変数を含んだ列を全て引く(ただし、定数/通常変数/スラック変数のみ)
        REP(i, SZ(A))if(is_eq[i] || b[i]<0){
            for(int j=0; j<=n_var+n_slk; ++j)
                a[n_row-1][j]-=a[i][j];
        }
        // 実行可能性チェック
        if(!__simplex(a, n_row, n_col, index) && a[n_row-1][0]!=0)
            return nullopt;
        // 2nd stageの目的関数行
        // 人為変数以外の列を0クリア
        for(int j=0;j<=n_var+n_slk;++j)
            a[n_row-1][j]=0;
        // 本来の目的関数の係数のマイナスをセット
        REP(j, n_var)
            a[n_row-1][1+j]=-c[j];
        // 目的関数から初期基底を除外する
        REP(i, n_row-1){
            T coeff;
            if(index[i]==0)
                continue;
            else if(index[i]<=n_var)
                coeff=c[index[i]-1];
            else if(index[i]<=n_var+n_slk)
                continue;
            else
                coeff=-1;
            for(int j=0;j<n_col-n_art;++j)
                a[n_row-1][j] += coeff * a[i][j];
        }
        return __simplex(a, n_row, n_col-n_art, index);
    }else{
        // 普通に解く
        REP(j, n_var)
            a[n_row-1][1+j]=-c[j];
        return __simplex(a, n_row, n_col, index);
    }
}

// 2分探索
template<class T>
int bsearch(const vector<T> &vec, T key, bool lower_bound=true){
    int left=0, right=SZ(vec)-1;
    while(left <= right){
        int mid = (left+right)/2;
        T val = vec[mid]; // ここを任意の関数に書き換えてよく使う
        if (val < key  // 降順の場合は不等号を逆にする
            || (!lower_bound && val == key))
            left=mid+1;
        else
            right=mid-1;
    }
    return left;
}

// 3分探索で最小値を与えるxを求める
double tsearch(function<double(double)> fn, double low, double high, double eps){
    while (abs(high-low)>eps) {
        double x1 = (low * 2 + high) / 3;
        double x2 = (low + high * 2) / 3;
        if (fn(x1) > fn(x2)) low = x1;
        else high = x2;
    }
    return low;
}

// ダイクストラ：始点sからその他頂点への最短距離を返す
template<class T>
vector<T> dijkstra(const vector< vector< pair<int, T> > > &adj, int n, int s){ // (隣接, ノード数, 始点)
    vector<T> dist(n, numeric_limits<T>::max()/2);
    priority_queue< pair<T, int>, vector< pair<T, int> >, greater< pair<T, int> > > pq;
    pq.emplace(0, s); // (コスト, ノード)
    dist[s] = 0;
    VB seen(n, false);
    while (!pq.empty()){
        int v = pq.top().second;
        pq.pop();
        if(seen[v]) continue;
        seen[v] = true;
        for (auto [to, cost] : adj[v]){ // ノード v に隣接しているノードに対して
            if (!seen[to] && dist[v] + cost < dist[to]){
                dist[to] = dist[v] + cost;
                pq.emplace(dist[to], to);
            }
        }
    }
    return dist;
}

// Bellman-Ford: 始点sからその他頂点への最短距離を返す(負の距離があってもよいがO(頂点数x辺の数)かかる)
template<class T>
vector<T> bellman_ford(const vector< vector< pair<int, T> > > &adj, int n, int s){
    vector<T> dist(n, numeric_limits<T>::max()/2);
    dist[s] = 0;
    REP(i, n-1){// 各頂点
        REP(from, n) for(const auto &[to, cost]: adj[from]){// 各辺
            if (dist[from] + cost < dist[to])
                dist[to] = dist[from] + cost;
        }
    }
    // 負の閉路チェック
    REP(from, n) for(const auto &[to, cost]: adj[from]){// 各辺
        if (dist[from] + cost < dist[to])
            return vector<T>();
    }
    return dist;
}

// Floyd–Warshall: 全ペアの最短経路
template<class T>
vector< vector<T> > floyd_warshall(const vector< vector<T> > &cost, int n) {// (移動コスト行列, 頂点数)
    vector< vector<T> > cost_min = cost;
    REP(i, n)
        cost_min[i][i]=0;
    REP(k, n)REP(i, n)REP(j, n)
        cost_min[i][j]=min(cost_min[i][j], cost_min[i][k]+cost_min[k][j]);
    return cost_min;
}

// A*アルゴリズム
// cost_map[i][j]: 座標(i,j)に訪問するとコストが発生する
template<class T>
pair<T, vector<PII>> a_star(const vector<vector<T>> &cost_map, int start_row, int start_col, int end_row, int end_col, bool start_cost){
    const vector<PII> directions{{0,1},{0,-1},{1,0},{-1,0}};
    const int row_max = SZ(cost_map);
    const int col_max = SZ(cost_map.front());

    // ヒューリスティック関数 h
    auto heuristic = [&](int i, int j){return abs(i-end_row)+abs(j-end_col);};
    // f=g+h
    vector<vector<T>> f_value(row_max, vector<T>(col_max, numeric_limits<T>::max()/2));

    auto to_id = [&](int i, int j){return i*col_max+j;};
    auto to_point = [&](int id){return PII{id/col_max, id%col_max};};
    // 親のID
    VVI parent(row_max, VI(col_max, -1));

    // NOTE: コスト負は壁を表すことにする
    REP(i, row_max)REP(j, col_max) {
        if (cost_map[i][j] < 0) 
            f_value[i][j]=-1;
    }

    // f, i, j
    MINPQ<tuple<T,int,int>> pq;
    f_value[start_row][start_col] = (start_cost ?  cost_map[start_row][start_col]: 0) + heuristic(start_row, start_col);
    pq.emplace(f_value[start_row][start_col], start_row, start_col);

    while (!pq.empty()) {
        auto [now_f, now_row, now_col]=pq.top();
        pq.pop();
        // ゴール
        if (now_row==end_row && now_col == end_col)
            break;
        if(f_value[now_row][now_col] < now_f)
            continue;
        // 現在の真のコスト
        T now_g = now_f - heuristic(now_row, now_col);
        // 次のノードへ
        for(auto [di,dj]: directions){
            int next_row=now_row+di, next_col=now_col+dj;
            if (0>next_row || next_row>=row_max || 0>next_col || next_col>=col_max)
                continue;
            T nextCost = now_g + cost_map[next_row][next_col] + heuristic(next_row, next_col);
            if(nextCost < f_value[next_row][next_col]){
                f_value[next_row][next_col]=nextCost;
                parent[next_row][next_col]=to_id(now_row, now_col);
                pq.emplace(nextCost, next_row, next_col);
            }
        }
    }

    vector<PII> path;
    int id=to_id(end_row, end_col);
    while(id >= 0){
        auto [r,c] = to_point(id);
        path.emplace_back(r,c);
        id=parent[r][c];
    }
    REVERSE(path);
    return {f_value[end_row][end_col] - heuristic(end_row, end_col), path};
}

// ダブリング: step個先の要素を求める
class Doubling{
private:
    VVI doubling;
public:
    Doubling(const VI &next, LL step){
        int n = SZ(next);
        int log_step = 1;
        while ((1LL << log_step) <= step) log_step++;
        // doubling[k][i]: i番目から 2^k 進んだ頂点
        doubling.assign(log_step, VI(n));
        REP(i, n)
            doubling[0][i] = next[i];
        // 前処理
        REP(k, log_step - 1){
            REP(i, n)
                doubling[k + 1][i] = doubling[k][doubling[k][i]];
        }
    }
    // 頂点startからstep回移動した頂点
    int reach(int start, LL step){
        int now = start;
        for (int k = 0; step > 0LL; k++) {
            if (step & 1LL)
                now = doubling[k][now];
            step >>= 1;
        }
        return now;
    }
};

// LCA: Lowest Common Ancestor, 木の最近共通祖先
class LCA {
private:
    VVI parent;  // parent[k][u]:= u の 2^k 先の親
    VI dist;     // root からの距離

    // 根からの距離と1つ先の頂点を求める
    void lca_dfs(const VVI &adj, int v, int prev, int d) {
        parent[0][v] = prev;
        dist[v] = d;
        for (int to : adj[v]) if (to != prev)
            lca_dfs(adj, to, v, d + 1);
    }
public:
    LCA(const VVI &adj, int root = 0) {
        int n = SZ(adj);
        int log_n = 1;
        while ((1 << log_n) < n) log_n++;
        parent.assign(log_n, VI(n, -1));
        dist.assign(n, -1);
        // parent[0]とdistを埋める
        lca_dfs(adj, root, -1, 0);
        // ダブリング
        REP(k, log_n - 1) {
            REP(v, n)
                parent[k + 1][v] = (parent[k][v] < 0)? -1 : parent[k][parent[k][v]];
        }
    }
    int query(int u, int v) {// uとvの最近共通祖先
        if (dist[u] < dist[v])// uの方が深い
            swap(u, v);
        int log_n = SZ(parent);
        // 深さを同じにする
        REP (k, log_n) {
            if (((dist[u] - dist[v]) >> k) & 1) 
                u = parent[k][u];
        }
        if (u == v) return u;
        // k個先を確認して共通祖先でないなら, k個先に移動
        RREP (k, log_n) {
            if (parent[k][u] != parent[k][v]) {
                u = parent[k][u];
                v = parent[k][v];
            }
        }
        return parent[0][u];
    }
    int get_dist(int u, int v) {// 2頂点間の距離
        return dist[u] + dist[v] - 2 * dist[query(u, v)];
    }
    bool is_on_path(int u, int v, int a) {// aがu-vのパス上にあるか判定
        return get_dist(u, a) + get_dist(a, v) == get_dist(u, v);
    }
};

// 集合のハッシュ化: T=uint64_t, uint32_t, ULL, unsigned intなど
template<class T>
class ZobristHashing {
    vector<T> h;
public:
    ZobristHashing(int size, mt19937& rand_engine) {
        uniform_int_distribution<T> uniform(0, numeric_limits<T>::max());
        h.reserve(size);
        REP(i, size)
            h.emplace_back(uniform(rand_engine));
    }
    T hash(const vector<T> &array) {
        T value = 0;
        for(auto e:array)
            value ^= h[e];
        return value;
    }
};


// 増加可能経路を探し, 増加分のフローを返す
// adj[i]: (j, capacity, 逆向き辺のindex)のリスト
template<class T>
T max_flow_impl(vector< vector< tuple<int, T, int> > >& adj, int v, int t, T f, VB &visit) {
    if (v == t) return f;
    visit[v] = true;
    for (auto& [to, cap, rev] : adj[v]) {
        if (!visit[to] && cap > 0) {
            T d = max_flow_impl(adj, to, t, min(f, cap), visit);
            if (d > 0) {
                cap -= d;
                T &rev_cap = get<1>(adj[to][rev]);
                rev_cap += d;
                return d;
            }
        }
    }
    return 0;
}
// Ford-Fullkerson法: s-t最大フロー
template<class T>
T max_flow(const vector< vector< pair<int, T> > >& adj, int n, int s, int t) {
    // adj_rev[i]: (j, capacity, 逆向き辺のindex)のリスト
    vector< vector< tuple<int, T, int> > > adj_rev(n);
    REP(i, n){
        for(const auto&[j, cap]: adj[i]){
            int size_i=SZ(adj_rev[i]), size_j=SZ(adj_rev[j]);
            adj_rev[i].emplace_back(j, cap, size_j);
            adj_rev[j].emplace_back(i, 0, size_i);// キャパゼロの逆向き辺
        }
    }

    T flow = 0, f = 0;
    VB visit(n, false);
    while ((f = max_flow_impl(adj_rev, s, t, numeric_limits<T>::max()/2, visit)) > 0) { // 増加分
        flow += f;
        fill(ALL(visit), false);
    }
    return flow;
}


// 最小費用流: s->t
template<class CAP, class COST>
COST min_cost_flow(const vector< vector< tuple<int, CAP, COST> > >& adj, int n, int s, int t, CAP init_flow) {
    // adj_rev[i]: (j, capacity, cost, 逆向き辺のindex)のリスト
    vector< vector< tuple<int, CAP, COST, int> > > adj_rev(n);
    REP(i, n){
        for(const auto&[j, cap, cost]: adj[i]){
            int size_i=SZ(adj_rev[i]), size_j=SZ(adj_rev[j]);
            adj_rev[i].emplace_back(j, cap, cost, size_j);
            adj_rev[j].emplace_back(i, 0, -cost, size_i);// キャパゼロの逆向き辺
        }
    }

    const COST COST_INF = numeric_limits<COST>::max()/2;
    vector<COST> dist(n);
    VI prev_vertex(n);
    VI prev_index(n);

    COST ans = 0;
    CAP flow = init_flow;
    while (flow > 0) {
        fill(ALL(dist), COST_INF);
        dist[s] = 0;
        while (true) {
            bool update = false;
            REP(v, n){
                if (dist[v] == COST_INF) continue;
                REP(i, SZ(adj_rev[v])) {
                    const auto &[to, cap, cost, rev] = adj_rev[v][i];
                    if (cap > 0 && dist[to] > dist[v] + cost) {
                        dist[to] = dist[v] + cost;
                        prev_vertex[to] = v;
                        prev_index[to] = i;
                        update = true;
                    }
                }
            }
            if (!update) break;
        }

        if (dist[t] == COST_INF) return COST_INF;

        CAP d = flow;
        for (int v = t; v != s; v = prev_vertex[v]) {
            const CAP& cap = get<1>(adj_rev[prev_vertex[v]][prev_index[v]]);
            d = min(d, cap);
        }
        flow -= d;
        ans += dist[t] * d;
        for (int v = t; v != s; v = prev_vertex[v]) {
            auto &[to, cap, cost, rev] = adj_rev[prev_vertex[v]][prev_index[v]];
            CAP &rev_cap = get<1>(adj_rev[to][rev]);
            cap -= d;
            rev_cap += d;
        }
    }
    return ans;
}

// 幅優先で訪問したノードの順番を返す
template<class T>
VI bfs(const vector< vector< pair<int, T> > > &adj, int n, int s){ // (隣接, ノード数, 始点)
    queue<int> que;
    que.emplace(s);
    VB seen(n, false);

    VI visit;
    while (!que.empty()){
        int v = que.front();
        que.pop();
        if (!seen[v])
            visit.emplace_back(v);
        seen[v] = true;
        for (auto [to, cost] : adj[v]){ // ノード v に隣接しているノードに対して
            if (!seen[to])
                que.emplace(to);
        }
    }
    return visit;
}

// 深さ優先で訪問したノードの順番を返す
template<class T>
VI dfs(const vector< vector< pair<int, T> > > &adj, int n, int s){  // (隣接, ノード数, 始点)
    stack<int> st;
    st.emplace(s);
    VB seen(n, false);

    VI visit;
    while (!st.empty()) {
        int v = st.top();
        st.pop();
        if (!seen[v])
            visit.emplace_back(v);
        seen[v] = true;
        for (auto [to, cost] : adj[v]){ // ノード v に隣接しているノードに対して
            if (!seen[to])
                st.emplace(to);
        }
    }
    return visit;
}

// 深さ優先（再帰）で訪問したノードの順番を返す
template<class T>
void dfs_recursive_impl(const vector< vector< pair<int, T> > > &adj, int s, VB &seen, VI &visit){
    if (!seen[s])
        visit.emplace_back(s);
    seen[s] = true;
    for (auto [to, cost] : adj[s]){
        if (!seen[to])
            dfs_recursive_impl(adj, to, seen, visit);
    }
}
template<class T>
VI dfs_recursive(const vector< vector< pair<int, T> > > &adj, int n, int s){
    VB seen(n, false);
    VI visit;
    dfs_recursive_impl(adj, s, seen, visit);
    return visit;
}

// 深さ優先で抜けるときにpushする
void sccd_impl(const VVI &adj, int s, VB &seen, VI &visit){
    seen[s] = true;
    EACH(to, adj[s])
        if (!seen[to])
            sccd_impl(adj, to, seen, visit);
    visit.emplace_back(s);
}
// Strongly Connected Component Decomposition (強連結成分分解)
VVI sccd(const VVI &adj, int n){
    // 1. DFSで順序付け
    VB seen(n, false);
    VI visit;
    REP(s, n){
        if (!seen[s])
            sccd_impl(adj, s, seen, visit);
    }
    // 2. 反転させたグラフをDFSの逆順でたどる
    VVI adj_rev(n);
    REP(i, n)EACH(j, adj[i])
        adj_rev[j].emplace_back(i);
    seen = VB(n, false);
    VVI connected;
    RITR(it, visit){
        if (seen[*it])
            continue;
        VI component;
        sccd_impl(adj_rev, *it, seen, component);
        connected.emplace_back(component);
    }
    return connected;
}

// 有向グラフの順序で見て, 頂点に順序をつける
VI topological_sort(const VVI &adj, int n){
    // 入次数
    VI indeg(n, 0);
    REP(i, n)EACH(j, adj[i])
        indeg[j]++;
    
    priority_queue< int, VI, greater<int> > heap;
    REP(i, n){
        if (indeg[i] == 0) 
            heap.emplace(i);
    }
    VI ans;
    while (!heap.empty()) {
        // 入次数がゼロで番号が若い順に訪問する
        int i = heap.top();
        heap.pop();
        ans.emplace_back(i);
        // iをグラフから削除
        EACH(j, adj[i]) {
            indeg[j]--;
            if (indeg[j] == 0)
                heap.emplace(j);
        }
    }
    return ans;
}

// ループを持つかどうか
bool contains_loop_impl(const VVI &adj, int s, VB &seen, int prev, bool direct){
    if (seen[s])
        return true;
    seen[s] = true;
    EACH(to, adj[s]){
        if(direct && prev==to) return true;// 有向グラフで頂点間を行き来できる場合
        if (prev!=to && contains_loop_impl(adj, to, seen, s, direct))
            return true;
    }
    return false;
}
bool contains_loop(const VVI &adj, int n, bool direct){// direct: 有向グラフならtrue
    VB seen(n, false);
    REP(i, n)
        if (!seen[i] && contains_loop_impl(adj, i, seen, -1, direct))
            return true;
    return false;
}

class UnionFind {
private:
    int n;
    VI parent; //親のノード情報、自身が根の場合はその集合のサイズを負で持つ
public:
    UnionFind(int n) : n(n), parent(n, -1){} // -1で初期化
    // データxが属する木の根
    int root(int x) {
        if (parent[x] < 0) return x;
        return parent[x]=root(parent[x]);
    }
    // xとyの木を併合
    void merge(int x, int y) {
        x = root(x);
        y = root(y);
        if (x == y) return;
        if (parent[x] > parent[y])
            swap(x, y);
        parent[x] += parent[y]; // 個数更新
        parent[y] = x; // 根の付け変え
    }
    // 同じ木に属するかどうか
    bool same(int x, int y) {
        return root(x) == root(y);
    }
    // xの集合のサイズ
    int size(int x) {
        return -parent[root(x)];
    }
    // xの属するメンバー
    VI members(int x) {
        int r = root(x);
        VI mem;
        REP(i, n)
            if (root(i) == r)
                mem.emplace_back(i);
        return mem;
    }
    // 根たち
    VI root_members() {
        VI mem;
        REP(i, n)
            if (parent[i] < 0)
                mem.emplace_back(i);
        return mem;
    }
    // 木の数
    int group_count() {
        return SZ(root_members());
    }
    // 根に対するすべてのメンバー
    map<int, VI> all_group_members() {
        map<int, VI> group_members;
        REP(member, n)
            group_members[root(member)].emplace_back(member);
        return group_members;
    }
    // 表示用
    friend ostream& operator<<(ostream& os, UnionFind& uf){
        os << uf.all_group_members();
        return os;
    }
};

// 最小全域木
// edges: {cost, v1, v2}
template<class T>
T kruskal(const vector< tuple<T,int,int> > &edges, int n) {
    // コストが小さい順にソート
    auto index = argsort(edges);
    UnionFind uf(n);
    T min_cost = 0;
    for(auto i: index){
        const auto &[cost, v1, v2] = edges[i];
        if (!uf.same(v1, v2)) {
            // 辺を追加しても閉路ができないなら、その辺を採用する
            min_cost += cost;
            uf.merge(v1, v2);
        }
    }
    return min_cost;
}

// プリム法
template<class T>
T prim(const vector<vector<pair<int, T>>> &adj, int n) {
    vector<T> mincost(n, numeric_limits<T>::max()/2);
    VB seen(n, false);
    T sum = 0;
    MINPQ<pair<T, int>> pq;
    // 0スタート
    pq.emplace(0, 0);
    while (!pq.empty()) {
        auto[v_cost,v]=pq.top();
        pq.pop();
        if(seen[v])continue;
        seen[v] = true;
        mincost[v]=v_cost;
        sum += v_cost;
        // 隣接頂点について最小コストを更新
        for(auto[to, to_cost]: adj[v]){
            if(!seen[to] && mincost[to]>to_cost){
                mincost[to]=to_cost;
                pq.emplace(to_cost, to);
            }
        }
    }
    return sum;
}

template <class T>
class SegmentTree {
public:
    enum class Mode{
        Maximum,
        Minimum,
        Summation,
    };
private:
    T default_val; // デフォルト値
    int num_leaves; // 葉の数 (2のべき)
    vector<T> data, lazy;
    VB upd; // 更新があるかどうかフラグ
    Mode mode;

    T default_value(){
        switch(mode){
            case Mode::Maximum:
                return numeric_limits<T>::min();
            case Mode::Minimum:
                return numeric_limits<T>::max();
            case Mode::Summation:
                return 0;
        }
        throw runtime_error("not support mode");
    }
    T op(T a, T b){
        switch(mode){
            case Mode::Maximum:
                return max(a, b);
            case Mode::Minimum:
                return min(a, b);
            case Mode::Summation:
                return a + b;
        }
        throw runtime_error("not support mode");
    }
    void parent_to_child(const T& parent, T& child){
        switch(mode){
            case Mode::Maximum:
            case Mode::Minimum:
                child = parent;
            CASE Mode::Summation:
                child += parent / 2;
        }
    }
    void lazy_to_data(T& l, T& d){
        switch(mode){
            case Mode::Maximum:
            case Mode::Minimum:
                d = l;
            CASE Mode::Summation:
                d += l;
        }
    }
    T to_lazy_value(int l, int r, T val){
        switch(mode){
            case Mode::Maximum:
            case Mode::Minimum:
                return val;
            CASE Mode::Summation:
                return (r-l) * val;
        }
        throw runtime_error("not support mode");
    }
    void update_impl(int a, int b, T val, int i, int l, int r) {
        eval(i);
        if (a <= l && r <= b) { // 範囲内のとき
            lazy[i] = to_lazy_value(l, r, val);
            upd[i] = true;
            eval(i);
        } else if (a < r && l < b) {
            update_impl(a, b, val, i * 2 + 1, l, (l + r) / 2);
            update_impl(a, b, val, i * 2 + 2, (l + r) / 2, r);
            data[i] = op(data[i * 2 + 1], data[i * 2 + 2]);
        }
    }
    T query_impl(int a, int b, int i, int l, int r) {
        eval(i);
        if (r <= a || b <= l) // 範囲外のとき
            return default_val;
        if (a <= l && r <= b)// 範囲内のとき
            return data[i];
        T vl = query_impl(a, b, i * 2 + 1, l, (l + r) / 2);
        T vr = query_impl(a, b, i * 2 + 2, (l + r) / 2, r);
        return op(vl, vr);
    }
    void eval(int i) {
        if (!upd[i]) return; // 更新が無ければスルー
        if (i < num_leaves - 1) { // 葉でなければ子に伝播
            parent_to_child(lazy[i], lazy[i * 2 + 1]);
            parent_to_child(lazy[i], lazy[i * 2 + 2]);
            upd[i * 2 + 1] = true;
            upd[i * 2 + 2] = true;
        }
        // 自身を更新
        lazy_to_data(lazy[i], data[i]);
        lazy[i] = default_val;
        upd[i] = false;
    }
public:
    SegmentTree(int n, Mode m) {// [0,n)の範囲を持つセグメント木
        int x = 1;
        while(n > x)
            x *= 2;
        mode = m;
        num_leaves = x;
        default_val = default_value();
        data.resize(num_leaves*2-1, default_val);
        lazy.resize(num_leaves*2-1, default_val);
        upd.resize(num_leaves*2-1, false);
    }
    void update(int a, int b, T val) {// [a,b)区間の値をvalに更新 (合計値の木ではvalを加算)
        update_impl(a, b, val, 0, 0, num_leaves);
    }
    void update(int i, T val) {// 値を更新 (合計値の木ではvalを加算)
        update_impl(i, i+1, val, 0, 0, num_leaves);
    }
    T query(int a, int b) {// [a,b)の最大値/最小値/合計値を取得
        return query_impl(a, b, 0, 0, num_leaves);
    }
    T query(int i) {// iの最大値/最小値/合計値を取得
        return query_impl(i, i+1, 0, 0, num_leaves);
    }
    // 表示用
    friend ostream& operator<<(ostream& os, const SegmentTree& st){
        int br=1;
        for (int i = 0; i < 2*st.num_leaves-1; ++i) {
            os << st.data[i];
            if (st.upd[i])
                os << "(" << st.lazy[i] << ")";
            os << ", ";
            if (br==i+1){
                os << endl;
                br = 2*br+1;
            }
        }
        return os;
    }
};

// Binary Indexed Tree
template <class T>
class BIT {
    int n;
    vector<T> bit;
public:
    BIT(int sz) : n(sz + 1), bit(n, 0) {}
    // a[i] += val ただし i in [0, n)
    void add(int i, T val) {
        for (int idx = i+1; idx <= n; idx += (idx & -idx))
            bit[idx] += val;
    }
    // a[0] + ... + a[i] ただし i in [0, n)
    T sum(int i) {
        T s(0);
        for (int idx = i+1; idx > 0; idx -= (idx & -idx)) 
            s += bit[idx];
        return s;
    }
};

// 座標圧縮
template<class T>
T compress(vector<T> &vec){
    map<T, T> mem;
    for(auto v: vec) mem[v] = 0;
    T sz = 0;
    for(auto &p: mem) p.second = sz++;
    for(auto &p: vec) p = mem[p];
    return sz;
}

// 配列を90度右回転させる
template<class T>
vector< vector<T> > rotate90(const vector< vector<T> >& mat){
    int m=mat.size(),n=mat[0].size();
    vector< vector<T> > ans(n, vector<T>(m));
    REP(i, m)REP(j, n)
        ans[j][i]=mat[m-1-i][j];
    return ans;
}

inline constexpr int DeBruijnBitPosition32[] = {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};
// 最下位ビットの1の位置
int lsb_pos32(unsigned int x) {
    if(x==0U) return -1;
    // 最下位ビットだけを残す
    x = x & -x;
    return DeBruijnBitPosition32[(x * 0x077CB531U) >> 27];
}
// 最上位ビットの1の位置
int msb_pos32(unsigned int x) {
    if(x==0U) return -1;
    // 最上位ビットだけを残す
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x = x ^ (x >> 1);
    return DeBruijnBitPosition32[(x * 0x077CB531U) >> 27];
}
// i bit目から上位ビット方向で最初に見つかる1の位置
int find_upper_pos32(unsigned int x, int i) {
    // i bit目未満をゼロに潰す
    const unsigned int mask = (-(1U << i)) | (1U << i);
    return lsb_pos32(x & mask);
}
// i bit目から下位ビット方向で最初に見つかる1の位置
int find_lower_pos32(unsigned int x, int i) {
    const unsigned int mask = ((1U << i) - 1U) | (1U << i);
    return msb_pos32(x & mask);
}

inline constexpr int DeBruijnBitPosition64 [] = {
    0, 1, 56, 2, 57, 49, 28, 3, 61, 58, 42, 50, 38, 29, 17, 4,
    62, 47, 59, 36, 45, 43, 51, 22, 53, 39, 33, 30, 24, 18, 12, 5,
    63, 55, 48, 27, 60, 41, 37, 16, 46, 35, 44, 21, 52, 32, 23, 11,
    54, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6,
};

int msb_pos64(ULL x){
    if (x == 0ULL) return -1;
    //最上位ビットだけを残す
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    x = x ^ (x >> 1);
    return DeBruijnBitPosition64[(x * 0x03f79d71b4ca8b09ULL) >> 58];
}
int lsb_pos64(ULL x){
    if (x == 0ULL) return -1;
    //最下位ビットだけを残す
    x = x & -x;
    return DeBruijnBitPosition64[(x * 0x03f79d71b4ca8b09ULL) >> 58];
}
// i bit目から上位ビット方向で最初に見つかる1の位置
int find_upper_pos64(ULL x, int i) {
    // i bit目未満をゼロに潰す
    const ULL mask = (-(1ULL << i)) | (1ULL << i);
    return lsb_pos64(x & mask);
}
// i bit目から下位ビット方向で最初に見つかる1の位置
int find_lower_pos64(ULL x, int i) {
    const ULL mask = ((1ULL << i) - 1ULL) | (1ULL << i);
    return msb_pos64(x & mask);
}

// 最長増加部分列の長さ
template<class T>
size_t LIS(const vector<T> &ary){
    vector<T> dp;// 長さkである増加部分列のうち, 最後の要素の最小値
    dp.reserve(ary.size());
    for(T a : ary){
        auto it = lower_bound(dp.begin(), dp.end(), a);
        if (it != dp.end())
            *it=a;
        else
            dp.emplace_back(a);
    }
    return dp.size();
}

// スライド最小値: ans[i]=min{ary[i],...,ary[i+k-1]} (0<=i<=n-k)
template<class T>
vector<T> slide_min(const vector<T> &ary, int k){
    vector<T> ans(ary.size()-k+1);
    deque<int> q;
    REP(i, SZ(ary)){
        // 現在の値以上を持つ場合削除
        while(!q.empty() && ary[q.back()] >= ary[i])
            q.pop_back();
        q.emplace_back(i);
        if (i-k+1 >= 0) {
            ans[i-k+1] = ary[q.front()];
            // ウィンドウから外れるため削除
            if (q.front() == i-k+1)
                q.pop_front();
        }
    }
    return ans;
}

template<LL MOD>
class ModInt {
    LL x;
public:
    constexpr ModInt(const LL &x = 0LL) {
        this->x = x % MOD;
        if(this->x < 0LL) this->x += MOD;
    }
    constexpr ModInt operator-() const { 
        return ModInt(-x);
    }
    constexpr ModInt& operator++() { 
        if ((x += 1LL) >= MOD) x -= MOD;
        return *this;
    }
    constexpr ModInt& operator--() { 
        if ((x -= 1LL) < 0LL) x += MOD;
        return *this;
    }
    constexpr ModInt operator++(int) { 
        ModInt tmp(*this);
        ++*this;
        return tmp;
    }
    constexpr ModInt operator--(int) { 
        ModInt tmp(*this);
        --*this;
        return tmp;
    }
    constexpr bool operator == (const ModInt& a) const {
        return x == a.x;
    }
    constexpr bool operator != (const ModInt& a) const {
        return x != a.x;
    }
    constexpr ModInt operator+(const ModInt& a) const {
        return ModInt(*this)+=a;
    }
    constexpr ModInt operator-(const ModInt& a) const {
        return ModInt(*this)-=a;
    }
    constexpr ModInt operator*(const ModInt& a) const {
        return ModInt(*this)*=a;
    }
    constexpr ModInt operator/(const ModInt& a) const {
        return ModInt(*this)/=a;
    }
    constexpr ModInt& operator+=(const ModInt& a) {
        if ((x += a.x) >= MOD) x -= MOD;
        return *this;
    }
    constexpr ModInt& operator-=(const ModInt& a) {
        if ((x -= a.x) < 0LL) x += MOD;
        return *this;
    }
    constexpr ModInt& operator*=(const ModInt& a) {
        x = (x * a.x) % MOD;
        return *this;
    }
    constexpr ModInt& operator/=(const ModInt& a) {
        return (*this) *= a.inv();
    }
    constexpr ModInt pow(LL n) const {
        ModInt ans(1LL), tmp(*this);
        while (n > 0) {
            if ((n & 1LL))
                ans *= tmp;
            tmp *= tmp;
            n >>= 1;
        }
        return ans;
    }
    constexpr ModInt inv() const {
        LL x0 = 1LL, x1 = 0LL, x2 = 0LL, y0 = 0LL, y1 = 1LL, y2 = 0LL, r0 = x, r1 = MOD, r2 = 0LL , q2 = 0LL;
        while (r1 != 0LL) {
            q2 = r0 / r1;
            r2 = r0 % r1;
            x2 = x0 - q2 * x1;
            y2 = y0 - q2 * y1;
            x0 = x1; x1 = x2;
            y0 = y1; y1 = y2;
            r0 = r1; r1 = r2;
        }
        assert (r0 == 1LL);
        return ModInt(x0);
        // return pow(MOD-2LL);
    }
    constexpr LL value() const {
        return x;
    }
    friend ostream& operator<<(ostream& os, const ModInt& a){
        os << a.x;
        return os;
    }
    constexpr friend bool operator == (int i, const ModInt& a) {
        return ModInt(i) == a;
    }
    constexpr friend bool operator != (int i, const ModInt& a) {
        return ModInt(i) != a;
    }
    constexpr friend ModInt operator+(int i, const ModInt& a) {
        return ModInt(i)+=a;
    }
    constexpr friend ModInt operator-(int i, const ModInt& a) {
        return ModInt(i)-=a;
    }
    constexpr friend ModInt operator*(int i, const ModInt& a) {
        return ModInt(i)*=a;
    }
    constexpr friend ModInt operator/(int i, const ModInt& a) {
        return ModInt(i)/=a;
    }
};

template<class T>
struct Matrix{
    int n,m;
    vector<T> data;
    Matrix():n(0),m(0){}
    Matrix(int n, int m, const T &val=0):n(n),m(m),data(n*m, val){}
    void init(int n, int m, const T &val=0){
        this->n=n, this->m=m;
        data.clear();
        data.resize(n*m, val);
    }
    T &at(int i, int j){
        assert(0<=i&&i<n);
        assert(0<=j&&j<m);
        return data[i*m+j];
    }
    const T &at(int i, int j) const {
        assert(0<=i&&i<n);
        assert(0<=j&&j<m);
        return data[i*m+j];
    }
    T &at(int i){
        assert(0<=i&&i<n*m);
        return data[i];
    }
    const T &at(int i) const {
        assert(0<=i&&i<n*m);
        return data[i];
    }
    friend ostream& operator<<(ostream& os, const Matrix<T> &mat){
        REP(i, mat.n){
            REP(j, mat.m)
                os << mat.at(i,j) << " ";
            os<<endl;
        }
        return os;
    }
    friend istream& operator>>(istream& is, Matrix<T> &mat){
        is >> mat.data;
        return is;
    }
};

// 三角行列（対角成分あり）
template<class T>
struct TriangularMatrix{
    int n;
    vector<T> data;
    TriangularMatrix():n(0){}
    TriangularMatrix(int n, const T &val=0):n(n),data(n*(n+1)/2, val){}
    void init(int n, const T &val=0){
        data.clear();
        data.resize(n*(n+1)/2, val);
    }
    T &at(int i, int j){
        if(i<j) swap(i,j);
        int base=i*(i+1)/2;
        return at(base+j);
    }
    const T &at(int i, int j) const {
        if(i<j) swap(i,j);
        int base=i*(i+1)/2;
        return at(base+j);
    }
    T &at(int i){
        assert(0<=i&&i<n*(n+1)/2);
        return data[i];
    }
    const T &at(int i) const {
        assert(0<=i&&i<n*(n+1)/2);
        return data[i];
    }
    friend ostream& operator<<(ostream& os, const TriangularMatrix<T> &mat){
        REP(i, mat.n){
            REP(j, i+1)
                os << mat.at(i,j) << " ";
            os<<endl;
        }
        return os;
    }
    friend istream& operator>>(istream& is, TriangularMatrix<T> &mat){
        is >> mat.data;
        return is;
    }
};

// tree / linked list node
// children: left_child -> .... -> right_child
// sibling: prev_sibling <- this -> next_sibling
template<class T>
struct Node{
    T data;
    Node<T>* parent;
    Node<T>* left_child;
    Node<T>* right_child;
    Node<T>* next_sibling;
    Node<T>* prev_sibling;

    Node(const T& data, Node<T>* parent=nullptr, 
            Node<T>* left_child=nullptr, Node<T>* right_child=nullptr, 
            Node<T>* next_sibling=nullptr, Node<T>* prev_sibling=nullptr){
        this->data=data;
        this->parent=parent;
        this->left_child=left_child;
        this->right_child=right_child;
        this->next_sibling=next_sibling;
        this->prev_sibling=prev_sibling;
    }

    Node<T>* add_child_on_left(const T &data){
        Node<T>* node = new Node<T>(data, this, nullptr, nullptr, this->left_child, nullptr);
        if(this->left_child!=nullptr)
            this->left_child->prev_sibling=node;

        this->left_child=node;
        if(this->right_child==nullptr)
            this->right_child=node;
        return node;
    }
    Node<T>* add_child_on_right(const T &data){
        Node<T>* node = new Node<T>(data, this, nullptr, nullptr, nullptr, this->right_child);
        if(this->right_child!=nullptr)
            this->right_child->next_sibling=node;

        this->right_child=node;
        if(this->left_child==nullptr)
            this->left_child=node;
        return node;
    }

    Node<T>* add_sibling_on_left(const T &data){
        Node<T>* node = new Node<T>(data, this->parent, nullptr, nullptr, this, this->prev_sibling);
        if(this->prev_sibling != nullptr)
            this->prev_sibling->next_sibling=node;
        else if(this->parent != nullptr)
            // 一番左にいる場合
            this->parent->left_child=node;
        this->prev_sibling=node;
        return node;
    }
    Node<T>* add_sibling_on_right(const T &data){
        Node<T>* node = new Node<T>(data, this->parent, nullptr, nullptr, this->next_sibling, this);
        if(this->next_sibling != nullptr)
            this->next_sibling->prev_sibling=node;
        else if(this->parent != nullptr)
            // 一番右にいる場合
            this->parent->right_child=node;
        this->next_sibling=node;
        return node;
    }
    static Node<T>* make_root(const T& data){
        return new Node(data);
    }
    static void delete_subtree(Node<T>* node){
        for(Node<T>* p=node->left_child, *next; p!=nullptr; p=next){
            next=p->next_sibling;
            delete_subtree(p);
        }

        if (node->prev_sibling!=nullptr)
            node->prev_sibling->next_sibling=node->next_sibling;
        else if(node->parent != nullptr)
            // 一番左にいる場合
            node->parent->left_child = node->next_sibling;

        if (node->next_sibling!=nullptr)
            node->next_sibling->prev_sibling=node->prev_sibling;
        else if(node->parent != nullptr)
            // 一番右にいる場合
            node->parent->right_child = node->prev_sibling;
        delete node;
    }
    void print(ostream& os=cout, bool print_sibling=true)const{
        if (print_sibling){
            if (this->parent != nullptr)
                os<<this->parent->data<<": ";
            else
                os<<"root: ";
            for(const Node<T>* p=this; p!=nullptr; p=p->next_sibling)
                os<<p->data<<" ";
            os<<endl;
        }
        for(const Node<T>* p=this->left_child; p!=nullptr; p=p->next_sibling)
            p->print(os, p==this->left_child);
    }
};

tuple<VI, VVD, VVI> kmeans(const VVD &points, int m, mt19937& rand_engine){
    assert(m>0);
    // サンプル数
    int n=SZ(points);
    assert(n>=m);
    // 次元数
    int dim = SZ(points.front());
    // セントロイド
    VVD centroid(m);
    {
        // 初期セントロイドを選ぶ
        VI candidates(n); ARANGE(candidates);
        VI selected; std::sample(candidates.begin(), candidates.end(), back_inserter(selected), m, rand_engine);
        REP(i, m)
            centroid[i]=points[selected[i]];
    }

    VI assign(n);
    VVI clusters(m);

    while(true){
        auto prev_assign=assign;
        REP(i, m)
            clusters[i].clear();
        // グループ分け
        REP(i, n){
            double min_dist=__DBL_MAX__;
            REP(j, m){
                // セントロイドとの距離
                double dist = 0;
                REP(k, dim){
                    double diff=centroid[j][k] - points[i][k];
                    dist += diff*diff;
                }
                dist = sqrt(dist);
                if(min_dist>dist){
                    assign[i]=j;
                    min_dist=dist;
                }
            }
            clusters[assign[i]].emplace_back(i);
        }
        // セントロイド更新
        REP(i, m){
            // 平均点を求める
            fill(ALL(centroid[i]), 0);
            for(int j: clusters[i]){
                REP(k, dim)
                    centroid[i][k] += points[j][k];
            }
            REP(k, dim)
                centroid[i][k] /= SZ(clusters[i]);
        }
        if(prev_assign==assign)
            break;
    }
    return {assign, centroid, clusters};
}

int main() {
    // 数学
    dump(cumsum(VI{1,2,3,4,5}, false))
    dump(cumsum(VI{1,2,3,4,5}, true))
    dump(combi(4, 2))
    dump(combi_mod(4, 2, 5))
    dump(pow_mod(5, 5, 100))
    dump(inverse(2LL, 1000000007LL, true))
    dump(inverse(2LL, 1000000007LL, false))
    dump(extend_euclid(2LL, 1000000007LL))
    dump(pfact(100))
    dump(divisor(100))
    dump(eratosthenes(100))
    dump(to_baseB(180, 16))
    dump(to_base10<int>("B4", 16))
    dump(ModInt<998244353>(5)/2)
    dump(5/ModInt<998244353>(2))
    auto fn=[](double x){return (x-3)*(x-3);};
    dump(tsearch(fn,0,10,1e-6))
    dump(sqrt_int(144))
    dump(sqrt_int(143))
    dump(sqrt_int(145))
    // グラフ
    istringstream iss1(
        "4 5 0\n" // 頂点数 辺の数 開始ノード
        "0 1 1\n" // s t c
        "0 2 4\n"
        "1 2 2\n"
        "2 3 1\n"
        "1 3 5\n"
    );
    int n1, m1, start;
    iss1 >> n1 >> m1 >> start;
    vector< vector<PII> > adj1(n1);
    REP(i, m1){
        int s, t, cost;
        iss1 >> s >> t >> cost;
        adj1[s].emplace_back(t, cost);
    }
    dump(adj1)
    dump(dijkstra(adj1, n1, start))
    dump(bellman_ford(adj1, n1, start))
    dump(bfs(adj1, n1, start))
    dump(dfs(adj1, n1, start))
    dump(dfs_recursive(adj1, n1, start))
    //
    istringstream iss2(
        "9 11 \n" // 頂点数 辺の数
        "0 1 \n" // s t
        "1 6 \n"
        "6 0 \n"
        "3 1 \n"
        "3 5 \n"
        "5 8 \n"
        "8 4 \n"
        "4 3 \n"
        "5 7 \n"
        "7 2 \n"
        "2 7 \n"
    );
    int n2, m2;
    iss2 >> n2 >> m2;
    VVI adj2(n2);
    REP(i, m2){
        int s, t;
        iss2 >> s >> t;
        adj2[s].emplace_back(t);
    }
    dump(adj2)
    dump(sccd(adj2, n2))
    // ループチェック
    dump(contains_loop(adj2, n2, false))
    // UnionFind
    UnionFind uf(5);
    uf.merge(1,2);
    uf.merge(0,4);
    dump(uf)
    //
    VVI mat={{1,2,3},{4,5,6}};
    dump(rotate90(mat))
    dump(LIS(VI{3, 1, 4, 1, 5, 9, 2, 6}))
    dump(bsearch(VI{1,1,2,2,2,4}, 2))
    dump(bsearch(VI{1,1,2,2,2,4}, 2, false))
    dump(bsearch(VI{1,1,2,2,2,4}, 8))
    dump(bsearch(VI{1,1,2,2,2,4}, 8, false))
    dump(slide_min(VI{1,4,2,6,3,5}, 3))
    // 
    SegmentTree<int> st(8, SegmentTree<int>::Mode::Maximum);
    st.update(1,5,10);
    dump(st)
    st.update(5,8,20);
    dump(st)
    dump(st.query(0,8))
    // tree
    Node<int>* root = Node<int>::make_root(0);
    Node<int>* one = root->add_child_on_left(1);
    Node<int>* two = root->add_child_on_left(2);
    Node<int>* three = root->add_child_on_right(3);
    two->add_child_on_right(4);
    Node<int>* five = three->add_child_on_right(5);
    five->add_sibling_on_right(6);
    Node<int>* seven = one->add_sibling_on_right(7);
    seven->add_child_on_left(8);
    seven->add_child_on_right(9);
    root->print(cout);
    Node<int>::delete_subtree(root);
    // as linked list
    root = new Node<int>(0);
    one = root->add_sibling_on_right(1);
    two = one->add_sibling_on_right(2);
    root->print(cout);
    Node<int>::delete_subtree(root);

    dump(split_str("a|bc||d", '|', true))
    dump(split_str("a|bc||d", '|', false))
    dump(split_str("|a|bc||d||", '|', true))
    dump(split_str("|a|bc||d||", '|', false))

    return 0;
}
