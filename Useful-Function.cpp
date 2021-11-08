#define _CRT_SECURE_NO_WARNINGS
#define gets gets_s
#include <uthash.h>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <string>
#include <stdbool.h>
#include <stdlib.h>
using namespace std;



/*										   函数与算法												*/


//1.gcd(最大公约数[辗转相除])	//含递归版
unsigned long long gcd(unsigned long long a, unsigned long long b)
{
    unsigned long long t;
    while (b != 0)
    {
        t = a % b;
        a = b;
        b = t;
    }
    return a;
}
//递归版1
int gcd(int m, int n)
{
	if (m > n)
		return gcd(m - n, n);
	else if (m < n)
		return gcd(m, n - m);
	else
		return m;
}
//递归版2
int gcd(int a, int b)
{
	if (a % b == 0) return b;
	return gcd(b, a % b);
}


//2.Prime(质数筛[欧拉筛法]+[某种巧妙筛法])
//[欧拉筛法]
int a[100000001], num = 0;//a用于存储素数,num表示质数个数
bool check[100000001];//用于判断1表示非素数，0表示素数
void Prime(int n)
{
    check[1] = 1;
    for (int i = 2; i <= n; i++)
    {
        if (!check[i]) a[++num] = i;//发现素数就记下来
        for (int j = 1; j <= num && i * a[j] <= n; j++)
        {
            check[a[j] * i] = 1;//标记
            if (i % a[j] == 0) break;//关键于此，保证每个合数只被筛一次
        }
    }
	for (int i = 1; i <= num; i++)  //可选项，输出质数
        printf("%d ", a[i]);
}
//[某种巧妙筛法]
int prime[100000001];//判断是否为素数
void Prime(int n)
{
	for (int i = 2; i <= n; i++)
		prime[i] = 1;
	for (int i = 2; i * i <= n; i++)
	{
		if(prime[i])
		{
			for (int j = i * i; j <= n; j += i)
				prime[j] = 0;		
		}
	}
}


//3.BinarySearch(二分搜索)  //ps:要求数列有序
int BinarySearch(int key,int a[],int len)
{
    int ret = -1;
    int left = 0;
    int right = len - 1;
    while (left <= right)	//l=1,r=len,l<=r
    {
        int mid = (right - left) / 2 + left;
        if (a[mid] == key)
        {
            ret = mid; break;
        }
        else if (a[mid] > key)
        {
            right = mid - 1;
        }
        else left = mid + 1;
    }
    return ret;
}
//cpp版
int BinarySearch(vector<int>& nums, int target)
{
	int left = 0, right = nums.size() - 1;
	while (left <= right)
	{
		int mid = ((right - left) >> 1) + left; // 防止溢出 等同于(left + right)/2
		if (nums[mid] == target) return mid;
		else if (nums[mid] < target) left = mid + 1;
		else right = mid - 1;
	}
	return -1;
}
//寻找目标值或插入位置-->找第一个>=target的下标
int BinaryInsert(int* nums, int numsSize, int target) 
{
	int left = 0, right = numsSize - 1, ans = numsSize;
	while (left <= right) 
	{
		int mid = ((right - left) >> 1) + left;
		if (target <= nums[mid]) 
		{
			ans = mid;
			right = mid - 1;
		}
		else 
		{
			left = mid + 1;
		}
	}
	return ans;
}
//cpp版
int searchInsert(vector<int>& nums, int target) 
{
	int left = 0, right = nums.size() - 1;
	while (left <= right)
	{
		int mid = ((right - left) >> 1) + left;
		if (nums[mid] == target) return mid;
		else if (nums[mid] < target) left = mid + 1;
		else right = mid - 1;
	}
	return right + 1;
}


//4.Add(高精度加法[字符串])
int trans1(char c)
{
	switch (c)
	{
	case '0':return 0;
	case '1':return 1;
	case '2':return 2;
	case '3':return 3;
	case '4':return 4;
	case '5':return 5;
	case '6':return 6;
	case '7':return 7;
	case '8':return 8;
	case '9':return 9;
	case 'a':return 10;
	case 'b':return 11;
	case 'c':return 12;
	case 'd':return 13;
	case 'e':return 14;
	case 'f':return 15;
	case 'g':return 16;
	case 'h':return 17;
	case 'i':return 18;
	case 'j':return 19;
	case 'k':return 20;
	case 'l':return 21;
	case 'm':return 22;
	case 'n':return 23;
	case 'o':return 24;
	case 'p':return 25;
	case 'q':return 26;
	case 'r':return 27;
	case 's':return 28;
	case 't':return 29;
	case 'u':return 30;
	case 'v':return 31;
	case 'w':return 32;
	case 'x':return 33;
	case 'y':return 34;
	case 'z':return 35;
	}
}
char trans2(int n)
{
	switch (n)
	{
	case 0:return '0';
	case 1:return '1';
	case 2:return '2';
	case 3:return '3';
	case 4:return '4';
	case 5:return '5';
	case 6:return '6';
	case 7:return '7';
	case 8:return '8';
	case 9:return '9';
	case 10:return 'a';
	case 11:return 'b';
	case 12:return 'c';
	case 13:return 'd';
	case 14:return 'e';
	case 15:return 'f';
	case 16:return 'g';
	case 17:return 'h';
	case 18:return 'i';
	case 19:return 'j';
	case 20:return 'k';
	case 21:return 'l';
	case 22:return 'm';
	case 23:return 'n';
	case 24:return 'o';
	case 25:return 'p';
	case 26:return 'q';
	case 27:return 'r';
	case 28:return 's';
	case 29:return 't';
	case 30:return 'u';
	case 31:return 'v';
	case 32:return 'w';
	case 33:return 'x';
	case 34:return 'y';
	case 35:return 'z';
	}
}
void Add(int n,char x1[], char x2[], char sum[])//n进制，x1+x2=sum
{
	int t = 1, check = 0;
	if (strlen(x1) > strlen(x2))
	{
		int p = strlen(x2);
		x2[strlen(x1)] = '\0';
		sum[strlen(x1)] = '\0';
		for (int i = strlen(x1) - 1; i >= 0; i--)
		{
			if ((p - t) >= 0) { x2[i] = x2[p - t]; t++;  }
			else { x2[i] = '0';  }
		}
	}
	if (strlen(x1) < strlen(x2))
	{
		int p = strlen(x1);
		x1[strlen(x2)] = '\0';
		sum[strlen(x2)] = '\0';
		for (int i = strlen(x2) - 1; i >= 0; i--)
		{
			if ((p - t) >= 0) { x1[i] = x1[p - t]; t++; }
			else x1[i] = '0';
		}
	}
	if (strlen(x1) == strlen(x2)) sum[strlen(x2)] = '\0';
	for (int i = strlen(x1) - 1; i >= 0; i--)
	{
		sum[i] = trans2((trans1(x1[i]) + trans1(x2[i]) + check) % n);
		if ((trans1(x1[i]) + trans1(x2[i]) + check) >= n) check = 1;
		else check = 0;
	}
	if (check)
	{
		int q = strlen(sum);
		sum[q + 1] = '\0';
		for (int i = q; i > 0; i--)
		{
			sum[i] = sum[i - 1];
		}
		sum[0] = '1';
	}
}


//5.fibonacci(斐波那契数列) ps:配合Add()使用
char x1[10001] = "1", x2[10001] = "2", sum[10001] = "3";
void fibonacci(int n)
{
	for (int k = 3; k < n; k++)
	{
		if (k % 2) strcpy(x1, sum);
		else strcpy(x2, sum);
		Add(10, x1, x2, sum);		
	}
	if (n == 0) cout << "0" << endl;
	else if (n == 1) cout << "1" << endl;
	else if (n == 2) cout << "2" << endl;
	else cout << sum << endl;
}


//6.quickSort(快排[优化版])
void swap(int& a, int& b)   //C++自带[iostream]
{
	int tmp = a;
	a = b;
	b = tmp;
}
void insertSort(int a[], int left, int right)
{
	for (int i = left + 1; i <= right; i++)
		for (int j = i; j > 0 && a[j] < a[j - 1]; j--)
			swap(a[j], a[j - 1]);
}
void quickSort(int a[], int left, int right)
{
	if (left >= right)
		return;
	if (right - left + 1 < 10)
	{
		insertSort(a, left, right);
		return;
	}
	srand((int)time(NULL));//随机数制种
	int i = left, j = right, k, flag = 0, pivot = rand() % (right - left + 1) + left;
	swap(a[left], a[pivot]);
	while (i < j)
	{
		while (j > i && a[j] >= a[left])
		{
			if (a[j] == a[left])
			{
				for (k = j - 1; k > i; k--)
					if (a[k] != a[j])
					{
						swap(a[k], a[j]);
						break;
					}
				if (k == i)
				{
					if (a[left] >= a[i])
						swap(a[left], a[i]);
					else
					{
						swap(a[i], a[j]);
						swap(a[left], a[i - 1]);
						i--;
						j--;
					}
					flag = 1;
					break;
				}
				else continue;
			}
			j--;
		}
		if (flag) break;
		while (i < j && a[i] <= a[left])
		{
			if (a[i] == a[left] && i != left)
			{
				for (k = i + 1; k < j; k++)
				{
					if (a[k] != a[i])
					{
						swap(a[k], a[i]);
						break;
					}
				}
				if (k == j)
				{
					swap(a[left], a[j]);
					flag = 1;
					break;
				}
				else continue;
			}
			i++;
		}
		if (flag) break;
		swap(a[i], (i == j) ? a[left] : a[j]);
	}
	quickSort(a, left, i - 1);
	quickSort(a, j + 1, right);
}


//7.quickPow(快速幂)
long long quickPow(int base,int p)
{
	if (p == 0) 
	{
		return 1;
	}
	else
	{
		long long temp;
		temp = quickPow(base, p / 2);
		if (p % 2) return temp * temp * base;
		else return temp * temp;
	}
}
//bit manipulation version
long long quickPow(int base, int p)
{
	long long ans = 1;
	while (p)
	{
		if (p & 1) ans *= base;	//if the current p ends with 1(binary)
		base *= base;	// base self-multiply
		p >>= 1;	// p right move for 1
	}
	return ans;
}


//8.singleHash(字符串哈希[单哈希，大质数||自然溢出])
#define base 233
#define mod 212370440130137957ll
unsigned long long singleHash(char s[])
{
	unsigned long long ans = 0;
	for (int i = 0; i < strlen(s); i++)
		ans = (ans * base + (unsigned long long)s[i]) % mod;
	//ans = ans * base + (unsigned long long)s[i] 也可以
	//这里不使用mod让它自然溢出，定义为ull的数在超过2^32的时候会自然溢出
	return ans;
}


//9.multiply(高精度乘法[字符串])
int a[50001], b[50001], i, x, len, j, c[50001];
void multiply(char a1[], char b1[])
{
	a[0] = strlen(a1); b[0] = strlen(b1);//计算长度
	for (i = 1; i <= a[0]; ++i)a[i] = a1[a[0] - i] - '0';//将字符串转换成数字
	for (i = 1; i <= b[0]; ++i)b[i] = b1[b[0] - i] - '0';
	for (i = 1; i <= a[0]; ++i)for (j = 1; j <= b[0]; ++j)c[i + j - 1] += a[i] * b[j];
	len = a[0] + b[0];   //按乘法原理进行高精乘
	for (i = 1; i < len; ++i) if (c[i] > 9) { c[i + 1] += c[i] / 10; c[i] %= 10; }//进位
	while (c[len] == 0 && len > 1) len--;//判断位数
	for (i = len; i >= 1; --i) cout << c[i];//输出
}


//10.miniQsort(低码量快排[还可以优化])
void swap(int& a, int& b)   //C++自带[iostream]
{
	int tmp = a;
	a = b;
	b = tmp;
}
void randomShuffle(int array[], int len)
{

	for (int i = 1; i < len; i++)
	{
		srand((int)time(NULL));//随机数制种
		int j = rand() % (i + 1);
		swap(array[i], array[j]);
	}
}
//或用C++自带的random_shuffle函数[algorithm],用法random_shuffle(a,a+n) or random_shuffle(a+1,a+n+1)
void miniQsort(int a[], int l, int r)
{
	randomShuffle(a, r); //random_shuffle(a,a+n) or random_shuffle(a+1,a+n+1)
	int mid = a[(l + r) / 2];  //应用二分思想
	int i = l, j = r;
	do {
		while (a[i] < mid) i++;  //查找左半部分比中间数大的数
		while (a[j] > mid) j--;  //查找右半部分比中间数小的数
		if (i <= j)   //如果有一组不满足排序条件（左小右大）的数
		{
			swap(a[i], a[j]);  //交换
			i++;
			j--;
		}
	} while (i <= j);
	if (l < j) miniQsort(a, l, j);  //递归搜索左半部分
	if (i < r) miniQsort(a, i, r);  //递归搜索右半部分
}


//11.read(快读)   ps:调用方法   变量名=read();
inline int read() //long long 也可
{
	register int s = 0, w = 1;//s是数值，w是符号 
	register char ch = getchar();
	while (ch < '0' || ch>'9')//将空格、换行与符号滤去 
	{
		if (ch == '-')//出现负号表示是负数 
		{
			w = -1;
			ch = getchar();//继续读入
		}
	}
	while (ch >= '0' && ch <= '9')//循环读取每一位的数字 
	{
		s = s * 10 + ch - '0';//将每一位的结果累加进s 
		//or s = (s<<1) + (s<<3) + (ch^48);   //使用位运算，更快
		ch = getchar();
	}
	return s * w;//乘上符号 
}
//更简单的iostream::sync_with_stdio(false);	放在main函数第一行,提升cin，cout速度
//执行完上述语句后只能使用{cin,cout}和{printf,scanf,putchar,getchar}的一边


//12.write(快写)  ps:调用方法 write(变量名);
inline void write(long long a)
{
	if (a >= 10) write(a / 10);
	putchar(a % 10 + '0');
}


//13.MaxSubseqSum(最大子列和[在线处理])
int MaxSubseqSum(int a[], int n)
{
	int ThisSum=0, MaxSum=a[0];
	for (int i = 0; i < n; i++)
	{
		ThisSum += a[i];  //向右累加
		if (ThisSum > MaxSum) MaxSum = ThisSum;  //发现更大和则更新当前结果
		else if (ThisSum < 0) ThisSum = 0;  //如果当前子列和为负，则不可能使后面的部分和增大，舍弃之

	}
	return MaxSum;
}


//14.myReverse(递归实现逆序输出字符串)
//读写版
void myReverse(void)
{
	char c = getchar();
	if (c != '\n')	myReverse();
	else return;
	putchar(c);
}
//已有字符串逆序输出
void myReverse(char *s)
{
	char c = *s;
	if (c != 0)	myReverse(++s);
	else return;
	putchar(c);
}


//15.cNum(求组合数函数，即Cji)
int cNum(int i, int j)  //返回组合数，亦即杨辉三角i行j列元素, i下j上
{
	if (j == 0) return 1;
	if (j == 1) return i;
	return cNum(i, j - 1) * (i - j + 1) / j;
}


//16.Move(实现数组左移或右移k位函数)
void Move(int arr[], int n, int k)
{
	k = k % n;  //n的整数倍移动相当于未移动
	//k = n - k;  加上此行代码即为右移
	for (int i = 0; i < n / 2; i++)
	{
		int t = arr[i];
		arr[i] = arr[n - 1 - i];
		arr[n - 1 - i] = t;
	}
	for (int i = n - k, j = 0; j < k / 2; i++, j++)
	{
		int t = arr[i];
		arr[i] = arr[n - 1 - j];
		arr[n - 1 - j] = t;
	}
	for (int i = 0; i < (n - k) / 2; i++)
	{
		int t = arr[i];
		arr[i] = arr[n - k - 1 - i];
		arr[n - k - 1 - i] = t;
	}
}


//17.IntToBinary(int型整数转换为二进制字符串)
char bin[40];
void IntToBinary(int n, char bin[])
{
	for (int i = 0, k = 0; i < 40; i++)
	{
		unsigned int mask = 1 << (31 - k);
		if (i % 5 - 4)
		{
			bin[i] = ((unsigned)(n & mask) >> (31 - k)) + '0';
			k++;
		}
		else bin[i] = ' ';
	}
	bin[39] = 0;
}


//18.delSubstr(删除字串)
int delSubstr(char str[], const char substr[])
{
	int p = 0, len = strlen(str);
	for (int i = 0; i < len; i++)
	{
		int j = 0;
		if (*(str + i) == *substr)
		{
			for (j = 0; j < strlen(substr); j++)
				if (*(str + i + j) != *(substr + j)) break;
			if (j == strlen(substr)) i += strlen(substr) - 1;
		}
		if (j != strlen(substr))
		{
			*(str + p) = *(str + i);
			p++;
		}
	}
	*(str + p) = 0;
	if (strlen(str) < len) return 1;
	else return 0;
}


//19.RemoveDuplicate(去掉字符串中重复字符)
int asc[256];   //空间换时间，标记是否已出现
void RemoveDuplicate(char* s)
{
	int r, w, i, len;
	len = strlen(s);
	for (r = w = 0; r < len; r++)
	{
		if (!asc[s[r]])
		{
			s[w++] = s[r];
			asc[s[r]]++;
		}
	}
	s[w] = 0;
}


//20.FileConnect(连接filename1、filename2 和 filename3 三个文件的内容到 filename 文件中)
void FileConnect(char* filename, char* filename1, char* filename2, char* filename3)
{
	FILE* c1, * c2, * c3, * c;
	c = fopen(filename, "w");
	c1 = fopen(filename1, "r");
	c2 = fopen(filename2, "r");
	c3 = fopen(filename3, "r");
	char a[101];
	while (fgets(a, 100, c1))
	{
		fputs(a, c);
	}
	while (fgets(a, 100, c2))
	{
		fputs(a, c);
	}
	while (fgets(a, 100, c3))
	{
		fputs(a, c);
	}
	fclose(c);
	fclose(c1);
	fclose(c2);
	fclose(c3);
}


//21.ArraySwap(二分段交换)
void ArraySwap(int arr[], int n, int k)
{
	int t;
	for (int i = 0; i < n / 2; i++)
	{
		t = arr[i];
		arr[i] = arr[n - 1 - i];
		arr[n - 1 - i] = t;
	}
	for (int i = n - k, j = 0; j < k / 2; i++, j++)
	{
		t = arr[i];
		arr[i] = arr[n - 1 - j];
		arr[n - 1 - j] = t;
	}
	for (int i = 0; i < (n - k) / 2; i++)
	{
		t = arr[i];
		arr[i] = arr[n - k - 1 - i];
		arr[n - k - 1 - i] = t;
	}
}


//22.BucketSort(简单桶排序，大小为n的整型数组a中元素满足：0≤a[i]≤k，时间复杂度O(n))
void BucketSort(int a[], int n, int k)
{
	int* check = (int*)malloc(sizeof(int) * (k + 1));
	for (int i = 0; i < k + 1; i++)
		check[i] = 0;
	for (int i = 0; i < n; i++)
		check[a[i]]++;
	int pos = 0;
	for (int i = 0; i < k + 1; i++)
		while (check[i])
		{
			a[pos++] = i;
			check[i]--;
		}
}


//23.findKthLargest(查找第k大的元素)
int partition(int* a, int l, int r) 
{
	int x = a[r], i = l - 1;
	for (int j = l; j < r; ++j) 
	{
		if (a[j] <= x) 
		{
			int t = a[++i];
			a[i] = a[j], a[j] = t;
		}
	}
	int t = a[i + 1];
	a[i + 1] = a[r], a[r] = t;
	return i + 1;
}
int quickSelect(int* a, int l, int r, int index) 
{
	int q = partition(a, l, r);
	if (q == index) 
	{
		return a[q];
	}
	else 
	{
		return q < index ? quickSelect(a, q + 1, r, index)
			: quickSelect(a, l, q - 1, index);
	}
}
int findKthLargest(int* nums, int numsSize, int k) 
{
	return quickSelect(nums, 0, numsSize - 1, numsSize - k);
}


//24.MergeSort(归并排序)
int min(int x, int y) 
{
	return x < y ? x : y;
}
//迭代版
void MergeSort(int arr[], int len) {
	int* a = arr;
	int* b = (int*)malloc(len * sizeof(int));
	int seg, start;
	for (seg = 1; seg < len; seg += seg) 
	{
		for (start = 0; start < len; start += seg + seg) 
		{
			int low = start, mid = min(start + seg, len), high = min(start + seg + seg, len);
			int k = low;
			int start1 = low, end1 = mid;
			int start2 = mid, end2 = high;
			while (start1 < end1 && start2 < end2)
				b[k++] = a[start1] < a[start2] ? a[start1++] : a[start2++];
			while (start1 < end1)
				b[k++] = a[start1++];
			while (start2 < end2)
				b[k++] = a[start2++];
		}
		int* temp = a;
		a = b;
		b = temp;
	}
	if (a != arr) 
	{
		int i;
		for (i = 0; i < len; i++)
			b[i] = a[i];
		b = a;
	}
	free(b);
}
//递归版
void merge_sort_recursive(int arr[], int reg[], int start, int end) {
	if (start >= end)
		return;
	int len = end - start, mid = (len >> 1) + start;
	int start1 = start, end1 = mid;
	int start2 = mid + 1, end2 = end;
	merge_sort_recursive(arr, reg, start1, end1);
	merge_sort_recursive(arr, reg, start2, end2);
	int k = start;
	while (start1 <= end1 && start2 <= end2)
		reg[k++] = arr[start1] < arr[start2] ? arr[start1++] : arr[start2++];
	while (start1 <= end1)
		reg[k++] = arr[start1++];
	while (start2 <= end2)
		reg[k++] = arr[start2++];
	for (k = start; k <= end; k++)
		arr[k] = reg[k];
}
void MergeSort(int arr[], const int len) {
	int *reg=(int*)malloc(sizeof(int)*len);
	merge_sort_recursive(arr, reg, 0, len - 1);
}


//25.Knapsack_01(01背包，DP问题)
//二维dp
int Knapsack_01(int num, int* weight, int* value, int bagweight)
{
	//申请二维数组
	//dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少
	int** dp = (int**)malloc(sizeof(int*) * (num + 1));
	for (int i = 0; i <= num; i++)
		dp[i] = (int*)malloc(sizeof(int) * (bagweight + 1));
	//memset(dp, 0, sizeof(dp));用memset()有问题，还不确定是否需要置0
	//初始化
	for (int j = weight[0]; j <= bagweight; j++)
		dp[0][j] = value[0];
	for (int i = 1; i < num; i++)//遍历物品
		for (int j = 0; j <= bagweight; j++)//遍历背包容量
			if (j < weight[i]) dp[i][j] = dp[i - 1][j];
			else dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
	return dp[num - 1][bagweight];
}
//一维dp
int Knapsack_01(int num, int* weight, int* value, int bagweight)
{
	//申请一维数组
	int* dp = (int*)malloc(sizeof(int) * (bagweight + 1));
	//初始化,dp[0]=0
	//价值都是正整数则非0下标都初始化为0，价值有负数，则非0下标就要初始化为负无穷
	memset(dp, 0, sizeof(int) * (bagweight + 1));
	for (int i = 0; i < num; i++)//遍历物品
		for (int j = bagweight; j >= weight[i]; j--)//遍历背包容量
			dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
	return dp[bagweight];
}
//bagweight过大时，无法申请dp数组(MLE)时处理方法
int Knapsack_01_Bigbag(int num, int* weight, int* value, int bagweight)
{
	int MinWeight = weight[1], SumWeight = weight[1];
	for (int i = 2; i <= num; i++)
	{
		if (weight[i] < MinWeight) MinWeight = weight[i];
		SumWeight += weight[i];
	}
	//在预处理时将每个物品的体积都减去（MinWeight-1）
	MinWeight--;
	for (int i = 1; i <= num; i++)
		weight[i] -= MinWeight;
	SumWeight -= num * MinWeight;
	int** dp = (int**)calloc(num + 1, sizeof(int*));
	for (int i = 0; i <= num; i++)
		dp[i] = (int*)calloc(SumWeight + 1, sizeof(int));
	//dp[k][j]表示选了k个,修改后的物品的重量为j，则当前的总体积为：j+k*MinWeight 
	for (int i = 1; i <= num; i++)
		for (int j = SumWeight; j >= weight[i]; j--)
			for (int k = 1; k <= num; k++)
				if (j + k * MinWeight <= bagweight)
					dp[k][j] = max(dp[k][j], dp[k - 1][j - weight[i]] + value[i]);
	int ans = 0;
	for (int i = 1; i <= num; i++)
		for (int j = 1; j <= SumWeight; j++)
			ans = max(ans, dp[i][j]);
	return ans;
}


//26.CompletePack(完全背包)
// 先遍历物品，再遍历背包
int CompletePack(int num, int* weight, int* value, int bagweight)
{
	int* dp = (int*)calloc(bagweight + 1, sizeof(int));
	for (int i = 0; i < num; i++)//遍历物品
		for (int j = weight[i]; j <= bagweight; j++)//遍历背包容量
			dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
	return dp[bagweight];
}
// 先遍历背包，再遍历物品
int CompletePack(int num, int* weight, int* value, int bagweight)
{
	int* dp = (int*)calloc(bagweight + 1, sizeof(int));
	for (int j = 0; j <= bagweight; j++)//遍历背包容量
		for (int i = 0; i < num; i++)//遍历物品
			if (j - weight[i] >= 0)
				dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
	return dp[bagweight];
}
// 背包容量bagweight变化版(eg.投资，债券，本金+利息)
int CompletePack_Variablebag(int num, int times, int* weight, int* value, int bagweight)
{
	int* dp = (int*)calloc(bagweight + 1, sizeof(int));
	for (int k = 0; k < times; k++)//刷新bagweight值
	{
		dp = (int*)realloc(dp, sizeof(int) * (bagweight + 1));
		for (int i = 0; i < num; i++)//遍历物品        
			for (int j = weight[i]; j <= bagweight; j++)//遍历背包容量            
				dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
		bagweight += dp[bagweight];
	}
	return bagweight;
}
// 最小花费背包
int CompletePack_Minimum(int num, int* weight, int* value, int bagweight)
{
	int* dp = (int*)calloc(bagweight + 1, sizeof(int));
	//dp[j]含义为花费为j时能获取的最大的价值
	for (int i = 1; i <= bagweight; i++)
		dp[i] = 0x3f3f3f3f;
	for (int i = 1; i <= num; i++)
	{
		for (int j = 1; j <= weight[i]; j++)
			dp[j] = min(dp[j], value[i]);
		//dp[0] == 0时，两个循环可交换
		for (int j = weight[i] + 1; j <= bagweight; j++)
			dp[j] = min(dp[j], dp[j - weight[i]] + value[i]);
	}
	return dp[bagweight];
}


//27.MultiplePack(多重背包)
int MultiplePack(int num, int* weight, int* value, int* nums, int bagweight)
{
	int* dp = (int*)calloc(bagweight + 1, sizeof(int));
	for (int i = 0; i < num; i++)//遍历物品
		for (int j = bagweight; j >= weight[i]; j--)//遍历背包容量
		//以上为01背包，然后加一个遍历个数
			for (int k = 1; k <= nums[i] && (j - k * weight[i]) >= 0; k++)
				dp[j] = max(dp[j], dp[j - k * weight[i]] + k * value[i]);
	return dp[bagweight];
}
//二维dp
long long MultiplePack_MaxChoice(int num, int* weight, int* nums, long long bagweight)
{
	//没有直接的bagweight限制时，将所有的weight累加起来，算出最多需要使用的Sumweight，当作bagweight
	long long** dp = (long long**)calloc(num + 1, sizeof(long long*));
	for (int i = 0; i <= num; i++)
		dp[i] = (long long*)calloc(bagweight + 1, sizeof(long long));
	//dp[i][j]表示从下标为[0-i]的物品里任意取放进容量为j的背包，选择种数最大是多少，即 Π (i物品选择个数)
	for (int i = 0; i <= num; i++)
		dp[i][0] = 1;
	for (int i = 1; i <= num; i++)	//遍历物品
		for (int j = bagweight; j >= 0; j--)	//遍历背包容量
		//以上为01背包，然后加一个遍历个数
			for (int k = 0; k <= nums[i] && k * weight[i] <= j; k++)
				dp[i][j] = max(dp[i][j], dp[i - 1][j - k * weight[i]] * k);
	return dp[num][bagweight];
}
//一维dp
long long MultiplePack_MaxChoice(int num, int* weight, int* nums, long long bagweight)
{
	//没有直接的bagweight限制时，将所有的weight累加起来，算出最多需要使用的Sumweight，当作bagweight
	long long* dp = (long long*)calloc(bagweight + 1, sizeof(long long));
	//dp[j]放进容量为j的背包，选择种数最大是多少，即 Π (i物品选择个数)
	dp[0] = 1;
	for (int i = 1; i <= num; i++)	//遍历物品
		for (int j = bagweight; j >= 0; j--)	//遍历背包容量
		//以上为01背包，然后加一个遍历个数
			for (int k = 0; k <= nums[i] && k * weight[i] <= j; k++)
				dp[j] = max(dp[j], dp[j - k * weight[i]] * k);
	return dp[bagweight];		
}
long long MultiplePack_TargetChoice_MinCost(int num, int* weight, int* nums, long long bagweight, long long target)
{
	//没有直接的bagweight限制时，将所有的weight累加起来，算出最多需要使用的Sumweight，当作bagweight
	long long* dp = (long long*)calloc(bagweight + 1, sizeof(long long));
	//dp[j]放进容量为j的背包，选择种数最大是多少，即 Π (i物品选择个数)
	dp[0] = 1;
	for (int i = 1; i <= num; i++)	//遍历物品
		for (int j = bagweight; j >= 0; j--)	//遍历背包容量
		//以上为01背包，然后加一个遍历个数
		//只带1个物品对Choice是没有任何帮助的，可以直接从2开始循环k，注意是否需要改动代码
			for (int k = 0; k <= nums[i] && k * weight[i] <= j; k++)
				dp[j] = max(dp[j], dp[j - k * weight[i]] * k);	
	long long ans = 0;
	//可用二分查找优化
	for (ans = 0; ans <= bagweight && dp[ans] < target; ans++);
	return ans;
}


//28.ApplyDimension_2Array(申请二维数组)
//利用二级指针申请
int** ApplyDimension_2Array(int m, int n)
{
	int** a = (int**)malloc(sizeof(int*) * m);
	for (int i = 0; i < m; i++)
		a[i] = (int*)malloc(sizeof(int) * n);
	return a;
}
void Withdraw(int** a,int m)
{
	for (int i = 0; i < m; i++)
		free(a[i]);
	free(a);
}
//已知列数，用数组指针形式申请
int(*ApplyDimension_2Array(int m))[3]
{
	int(*a)[3] = (int(*)[3])malloc(sizeof(int) * m * 3);
	return a;
}
void Withdraw(int(*a)[3])
{
	free(a);
}


//29.ShellSort(希尔排序,Sedgewick增量序列)
typedef int ElemType;
void ShellSort(ElemType a[], int n)
{
	int Si, i, j, k;
	ElemType temp;
	int Sedgewick[] = { 1073643521,603906049,268386305,150958081,67084289,37730305,16764929,9427969,4188161,2354689,1045505,587521,260609,146305,64769,36289,16001,8929,3905,2161,929,505,209,109,41,19,5,1,0 };
	for (Si = 0; Sedgewick[Si] >= n; Si++);
	for (i = Sedgewick[Si]; i > 0; i = Sedgewick[++Si])
		for (j = i; j < n; j++)
		{
			temp = a[j];
			for (k = j; k >= i && a[k - i] > temp; k -= i)
				a[k] = a[k - i];
			a[k] = temp;
		}
}


//30.HeapSort(堆排序)
void HeapAdjust(int* arr, int l, int r)
{
	int k = arr[l];
	for (int j = 2 * l; j <= r; j *= 2)
	{
		if (j<r && arr[j]>arr[j + 1]) j++;
		if (k <= arr[j]) break;
		arr[l] = arr[j];
		l = j;
	}
	arr[l] = k;
}
void HeapSort(int* arr, int len)
{
	for (int i = len / 2 - 1; i >= 0; i--)
		HeapAdjust(arr, i, len - 1);
	for (int i = len - 1; i > 0; i--)
	{
		swap(arr[0], arr[i]);
		HeapAdjust(arr, 0, i - 1);
	}
}


//31.RadixSort(基数排序)
int MaxBit(int data[], int n) //辅助函数，求数据的最大位数
{
	int maxData = data[0];
	for (int i = 1; i < n; i++)
	{
		if (maxData < data[i])
			maxData = data[i];
	}
	int d = 1;
	while (maxData >= 10)
	{
		maxData /= 10;
		d++;
	}
	return d;
}
void RadixSort(int data[], int n) //基数排序
{
	int maxbit = MaxBit(data, n);
	int* tmp = (int*)calloc(n, sizeof(int));
	int* count = (int*)calloc(10, sizeof(int)); //计数器
	int k;
	int radix = 1;
	for (int i = 1; i <= maxbit; i++) //进行d次排序
	{
		for (int j = 0; j < 10; j++)
			count[j] = 0; //每次分配前清空计数器
		for (int j = 0; j < n; j++)
		{
			k = (data[j] / radix) % 10; //统计每个桶中的记录数
			count[k]++;
		}
		for (int j = 1; j < 10; j++)
			count[j] = count[j - 1] + count[j]; //将tmp中的位置依次分配给每个桶
		for (int j = n - 1; j >= 0; j--) //将所有桶中记录依次收集到tmp中
		{
			k = (data[j] / radix) % 10;
			tmp[count[k] - 1] = data[j];
			count[k]--;
		}
		for (int j = 0; j < n; j++) //将临时数组的内容复制到data中
			data[j] = tmp[j];
		radix = radix * 10;
	}
	free(tmp);
	free(count);
}


//32.Fisher_Yates_Shuffle(洗牌算法，生成乱序数列)
void Fisher_Yates_Shuffle(int* array, int len)
{
	int i = len;
	int j = 0;
	int temp;
	if (i == 0)
		return;
	while (--i)
	{
		j = rand() % (i + 1);
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}


//33.RemoveElement(原地删除指定元素，返回新长度，双指针法)
int RemoveElement(int* nums, int numsSize, int val) 
{
	int left = 0;
	for (int right = 0; right < numsSize; right++) 
	{
		if (nums[right] != val)
		{
			nums[left] = nums[right];
			left++;
		}
	}
	return left;
}
//优化版,元素顺序改变
int removeElement(int* nums, int numsSize, int val) 
{
	int left = 0, right = numsSize;
	while (left < right) 
	{
		if (nums[left] == val) 
		{
			nums[left] = nums[right - 1];
			right--;
		}
		else left++;
	}
	return left;
}


//34.FullUse_Solution(计算用完背包空间方法数)
int FullUse_Solution(int num, int* weight, int bagweight)
{
	int** dp = (int**)calloc(num + 1, sizeof(int));
	for (int i = 0; i <= num; i++)
		dp[i] = (int*)calloc(bagweight + 1, sizeof(int));
	//dp[i][j]表示用1->i件物品用光j体积的办法总数(从1开始计数)
	for (int i = 1; i <= num; i++)
		for (int j = 1; j <= bagweight; j++)
		{
			if (j == weight[i]) dp[i][j] = dp[i - 1][j] + 1;
			else if (j > weight[i]) dp[i][j] = dp[i - 1][j] + dp[i - 1][j - weight[i]];
			else dp[i][j] = dp[i - 1][j];
		}
	return dp[num][bagweight];
}
//一维dp
int FullUse_Solution(int num, int* weight, int bagweight)
{
	int* dp = (int*)calloc(num + 1, sizeof(int));
	//dp[j]表示用完j空间方法数
	dp[0] = 1;//没有空间就不选，即一种方法
	for (int i = 1; i <= num; i++)
		for (int j = bagweight; j >= weight[i]; j--)
			dp[j] += dp[j - weight[i]];//现在的方法数+=我不选物品i时方法数
	return dp[bagweight];
}


//35.MinSubArrayLen(找出数组中满足其和≥target的长度最小的连续子数组,返回其长度)
//滑动窗口(O(n),O(1))
int MinSubArrayLen(int target, int* nums, int numsSize)
{
	if (!numsSize) return 0;
	int ans = INT_MAX;
	int start = 0, end = 0;
	int sum = 0;
	while (end < numsSize)
	{
		sum += nums[end];
		while (sum >= target)
		{
			ans = min(ans, end - start + 1);
			sum -= nums[start];
			start++;
		}
		end++;
	}
	return ans == INT_MAX ? 0 : ans;
}
//前缀和+二分查找(O(nlogn),O(n))
int lower_bound(int* a, int l, int r, int value)//返回有序序列第一个不小于value的位置
{
	if (a[r] < value) return -1;
	while (l < r)
	{
		int mid = (l + r) >> 1;
		if (a[mid] >= value)
		{
			r = mid;
		}
		else
		{
			l = mid + 1;
		}
	}
	return l;
}
int MinSubArrayLen(int target, int* nums, int numsSize)
{
	if (numsSize == 0)
		return 0;
	int ans = INT_MAX;
	int* sums = (int*)malloc(sizeof(int) * (numsSize + 1));
	// 为了方便计算，令 size = n + 1
	// sums[0] = 0 意味着前 0 个元素的前缀和为 0
	// sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
	// 以此类推
	for (int i = 1; i <= numsSize; i++)
	{
		sums[i] = sums[i - 1] + nums[i - 1];
	}
	for (int i = 1; i <= numsSize; i++)
	{
		int s = target + sums[i - 1];
		int bound = lower_bound(sums, 1, numsSize, s);
		if (bound != -1)
		{
			ans = min(ans, bound - (i - 1));
		}
	}
	return ans == INT_MAX ? 0 : ans;
}


//36.#min & #max(宏定义min()与max())
#define min(a, b)  ((a) < (b) ? (a) : (b))
#define max(a, b)  ((a) > (b) ? (a) : (b))


//37.swap(交换函数)
void swap(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}
//位运算
void swap(int* a, int* b)
{
	*a ^= *b;
	*b ^= *a;
	*a ^= *b;
}


//38.KMP(字符串匹配)
void GetNext(int* next, char* substr)
{
	int len = strlen(substr);
	int t1 = 0, t2 = -1;
	next[0] = -1;
	while (t1 < len)
	{
		if (t2 == -1 || substr[t1] == substr[t2])
			next[++t1] = ++t2;
		else t2 = next[t2];
	}
}
int KMP(char* str, char* substr)
{
	if (!substr[0]) return 0;	//为空串时,返回0,与C语言的strstr()以及Java的indexOf()定义相符
	int t1 = 0, t2 = 0, len = strlen(str), sublen = strlen(substr);
	int* next = (int*)calloc(sublen + 1, sizeof(int));	//注意+1
	GetNext(next, substr);
	while (t1 < len)
	{
		if (t2 == -1 || str[t1] == substr[t2])
			t1++, t2++;
		else t2 = next[t2];
		if (t2 == sublen) return t1 - sublen;	//此为数组下标,若为实际index,则+1
	}
	return -1;
}


//39.BMSearch(BM算法，字符串匹配)
int* MakeBCtable(char* substr)//根据坏字符规则做预处理，建立一张坏字符表
{
	int sublen = strlen(substr);
	//为建立坏字符表，申请256个空间
	int* skip = (int*)calloc(256, sizeof(int));
	//初始化
	for (int i = 0; i < 256; ++i)
		skip[i] = sublen;
	//给表中需要赋值的单元赋值，不在模式串中出现的字符就不用再赋值了
	for (int i = 0; i < sublen; i++)
		skip[substr[i]] = i;
	return skip;
}
int* MakeMStable(char* substr) {
	int sublen = strlen(substr);
	int* MS = (int*)calloc(sublen, sizeof(int));
	MS[sublen - 1] = sublen; //串与自身一定匹配
	int lower = sublen - 1, upper = sublen - 1;
	for (int i = upper - 1; i >= 0; i--) {
		if (i > lower && MS[sublen - upper + i - 1] <= i - lower)
			MS[i] = sublen - upper + i - 1;
		else
		{
			upper = i;
			lower = min(upper, lower);
			while (lower >= 0 && substr[lower] == substr[sublen - upper + lower - 1]) lower++;
			MS[i] = upper - lower;
		}
	}
	return MS;
}
int* MakeGStable(char* substr)
{
	int* MS = MakeMStable(substr);
	int sublen = strlen(substr);
	int* GS = (int*)calloc(sublen, sizeof(int));
	for (int i = 0; i < sublen; i++)
		GS[i] = sublen;
	for (int i = 0, j = sublen - 1; j >= 0; j--)
	{
		if (j + 1 == MS[j])
			while (i < sublen - j - 1)
				GS[i++] = sublen - j - 1;

	}
	for (int i = 0; i < sublen - 1; i++)
		GS[sublen - MS[i] - 1] = sublen - i - 1;
	free(MS);
	return GS;
}
int BMSearch(char* str, char* substr)
{
	int i = 0, sublen = strlen(substr), len = strlen(str), k = sublen - 1;
	if (!sublen) return 0;
	if (!len) return -1;
	int* BC = MakeBCtable(substr);
	int* GS = MakeGStable(substr);
	while (i <= len - sublen)
	{
		while (str[i + k] == substr[k])
			if (--k < 0)
				break;
		if (k < 0)
			break;
		else
			i += max(GS[k], k - BC[str[i + k]]);
	}
	free(BC);
	free(GS);
	if (i <= len - sublen) return i;
	else return -1; //匹配失败
}
//BMH算法,对BM算法的改进算法
//在字符串搜索过程中，遇到坏字符的概率要远远大于好后缀的情况
//所以在实际使用时，只使用坏字符表也有很好的效率
int BMHSearch(char* str, char* substr) {
	int* BC = MakeBCtable(substr);
	int i = 0, sublen = strlen(substr), len = strlen(str), k = sublen - 1;
	while (i <= len - sublen)
	{
		while (str[i + k] == substr[k])
			if (--k < 0)
				break;
		if (k < 0)
			break;
		else
			i += max(1, k - BC[str[i + k]]);
	}
	free(BC);
	if (i <= len - sublen) return i;
	else return -1; //匹配失败
}


//40.SundaySearch(简易但普遍高效的字符串匹配)
int SundaySearch(char* str, char* substr)
{
	int next[256];
	int len = strlen(str), sublen = strlen(substr);
	for (int i = 0; i < 256; ++i)
		next[i] = -1;
	for (int i = 0; i < sublen; ++i)
		next[substr[i]] = i;
	int strPos;  // s 的下标
	int substrPos;  // p 的下标
	int i = 0;
	while (i <= len - sublen)
	{
		strPos = i;
		substrPos = 0;
		while (strPos < len && substrPos < sublen && str[strPos] == substr[substrPos])
			strPos++, substrPos++;
		if (substrPos == sublen)
		{
			return i;
		}
		if (i < len - sublen)
			i += (sublen - next[str[i + sublen]]);
		else
			break;
	}
	return -1;
}


//41.Rabin_Karp_Search(Rabin-Karp 算法, 哈希用于字符串匹配)
#define Base 100000
//31这个数字是静置转换中常用的基数
#define PowBase 31
int Rabin_Karp_Search(char* str, char* substr)
{
	int len = strlen(str), sublen = strlen(substr);
	if (!sublen) return 0;
	if (!len) return -1;
	//31^sublen
	int power = 1;
	for (int i = 0; i < sublen; ++i)
		power = (power * PowBase) % Base;	//取余数为了控制power大小,不超过int最大数值
	//获取模式串的hashcode
	//eg.hash("abc") = a*31^2+b*31^1+c*31^0;
	int substrHash = 0;
	for (int i = 0; i < sublen; ++i)
		substrHash = (substrHash * PowBase + substr[i]) % Base;
	int strHash = 0;
	for (int i = 0; i < len; ++i)
	{
		strHash = (strHash * PowBase + str[i]) % Base;
		if (i < sublen - 1) continue;
		if (i >= sublen)
		{
			//abcd -a 移除第一位  减法 hashcode - a*31^m  m正好为目标字符串的长度
			strHash = (strHash - str[i - sublen] * power) % Base;
			//对负值进行处理
			if (strHash < 0)
			{
				//因为我们是对Base进行取余，故加上一次Base后，strHash必为正值
				strHash += Base;
			}
		}
		//此时 i = sublen - 1,即strHash与substrHash的串长相同
		//hashcode相等但字符顺序可能不同（例如"abc"与"bca"），因此需要二次检查
		if (strHash == substrHash)
		{
			int startIndex = i - sublen + 1;
			if (!strncmp(str + startIndex, substr, sublen))
				return startIndex;
		}
	}
	return -1;
}


//42.Mid(求两数平均值)
int Mid(int low, int high)
{
	return (high - low) / 2 + low;
	// OR (high - low >> 1) + low
	//等价于(high + low) / 2,但hight+low就可能会溢出
	//若取中间值靠右,(high - low + 1) / 2 + low
	// OR (high - low + 1 >> 1) + low
}


//43.IntSqrt(二分查找int型开根)
int IntSqrt(int x) 
{
	int left = 0, right = x, ans = 0;
	while (left <= right)
	{
		int mid = ((right - left) >> 1) + left;
		if ((long long)mid * mid <= x)
		{
			ans = mid;
			left = mid + 1;
		}
		else right = mid - 1;
	}
	return ans;
}


//44.CountOnesInBinary(计算二进制比特串内1的个数)
int CountOnesInBinary(int x) 
{
	//Brian Kernighan 算法
	//对于任意整数 x，令 x = x & (x-1)，该运算将 x 的二进制表示的最后一个 1 变成 0
	//因此，对 x 重复该操作，直到 x 变成 0，则操作次数即为 x 的「一比特数」
	//时间复杂度：O(logn)
	int ones = 0;
	while (x > 0) 
	{
		x &= (x - 1);
		ones++;
	}
	return ones;
}








/*										   类与结构体												*/


//1.SqList(动态分配顺序表)
#define LIST_INIT_SIZE 100   //存储空间初始分配量(可修改)
#define LISTINCREMENT 10   //存储空间分配增量(可修改)
typedef int ElemType;
class SqList
{
private:
	ElemType* elem;   //存储空间基地址，相当于数组首元素地址
	int length;   //表长，即元素个数
	int listsize;   //当前分配的存储容量(以sizeof(ElemType)为单位)
public:
	SqList() :elem(NULL), length(0), listsize(0)
	{}
	~SqList()
	{}
	int InitSqList()   //构造一个空的线性表L
	{
		if (elem)  return -1;  //线性表L若已存在，返回-1
		elem = (ElemType*)malloc(LIST_INIT_SIZE * sizeof(ElemType));
		if (!elem) return 0;   //存储分配失败
		listsize = LIST_INIT_SIZE;
		return 1;
	}
	void ListInput()
	{
		int n;
		cout << "请输入元素个数：";
		scanf("%d", &n);
		cout << "请输入元素：" << endl;
		if (n > listsize)
		{
			elem = (ElemType*)malloc(n * sizeof(ElemType));
			listsize = n;
		}
		for (int i = 0; i < n; i++)
			cin >> elem[i];
		length = n;
	}
	void print()
	{
		for (int i = 0; i < length; i++)
			cout << elem[i] << " ";
		putchar('\n');
	}
	int Insert(int i, ElemType e)    //指定位置插值函数，使用引用参数
	{
		if (i<1 || i>length + 1)   //i的合法取值为1至n+1
			return 0;
		if (length == listsize)   //溢出时扩充
		{
			ElemType* newbase;   //创建新地址
			newbase = (ElemType*)realloc(elem, (listsize + LISTINCREMENT) * sizeof(ElemType));
			if (newbase == NULL) return 0;   //扩充失败
			elem = newbase;   //更新基地址
			listsize += LISTINCREMENT;
		}
		for (int j = length - 1; j >= i - 1; j--)   //向后移动元素，空出第i个元素的位置elem[i-1]
			elem[j + 1] = elem[j];
		elem[i - 1] = e;
		length++;
		return 1;
	}
	int DeleteElement(int i)
	{
		if (i<1 || i>length) return 0;
		for (int j = i; j <= length - 1; j++)
			elem[j - 1] = elem[j];
		length--;
		return 1;
	}
	void MergeList(SqList* La, SqList* Lb)//已知顺序表La，Lb的元素按值非递减排列，归并La，Lb得到Lc，Lc也非递减
	{
		ElemType* pa = La->elem, * pb = Lb->elem;
		listsize = length = La->length + Lb->length;
		ElemType* pc = elem = (ElemType*)malloc(listsize * sizeof(ElemType));
		if (!elem) exit(1); //存储分配失败
		ElemType* pa_last = La->elem + La->length - 1, * pb_last = Lb->elem + Lb->length - 1;
		while (pa <= pa_last && pb <= pb_last)  //归并
		{
			if (*pa <= *pb) *pc++ = *pa++;
			else *pc++ = *pb++;
		}
		while (pa <= pa_last) *pc++ = *pa++;  //插入La中剩余元素
		while (pb <= pb_last) *pc++ = *pb++;  //插入Lb中剩余元素
	}
	void TriSqListOr(SqList B, SqList C)//A，B，C为3个递增有序的线性表，删除A中与B或C中相同的元素
	{
		int pa = 0, pb = 0, pc = 0;
		while ((pb <= B.length - 1) && (pc <= C.length - 1))
		{
			if (B.elem[pb] == elem[pa])
			{
				DeleteElement(pa + 1); pb++;
			}
			if (C.elem[pc] == elem[pa])
			{
				DeleteElement(pa + 1); pc++;
			}
			if (elem[pa] < B.elem[pb] && elem[pa] < C.elem[pc])
			{
				pa++;
				if (pa == length) return;
			}
			else if (elem[pa] > B.elem[pb]) pb++;
			else if (elem[pa] > C.elem[pc]) pc++;
		}
		if (pb == B.length)
		{
			while (pc <= C.length - 1)
				if (C.elem[pc] == elem[pa])
				{
					DeleteElement(pa + 1); pc++;
				}
				else if (elem[pa] < C.elem[pc])
				{
					pa++;
					if (pa == length) return;
				}
				else pc++;
		}
		else if (pc == C.length)
		{
			while (pb <= B.length - 1)
				if (B.elem[pb] == elem[pa])
				{
					DeleteElement(pa + 1); pb++;
				}
				else if (elem[pa] < C.elem[pb])
				{
					pa++;
					if (pa == length) return;
				}
				else pb++;
		}
	}
	void TriSqListAnd(SqList B, SqList C)//A，B，C为3个递增有序的线性表，删除A中与B且C中相同的元素
	{
		int pa = 0, pb = 0, pc = 0;
		while ((pb <= B.length - 1) && (pc <= C.length - 1))
		{
			if ((B.elem[pb] == elem[pa]) && (C.elem[pc] == elem[pa]))
			{
				pb++; pc++;
				DeleteElement(pa + 1);
			}
			if (elem[pa] < B.elem[pb] || elem[pa] < C.elem[pc])
			{
				pa++;
				if (pa == length) return;
			}
			else if (elem[pa] > B.elem[pb]) pb++;
			else if (elem[pa] > C.elem[pc]) pc++;
		}
	}
};


//2.Node(单链表)
typedef int ElemType;
#define LENG sizeof(Node)  //结点所占单元数
class Node
{
private:
	ElemType data;
	Node* next;
public:
	Node() :data(0), next(NULL)
	{}
	~Node()
	{}
	void Create()
	{
		Node* tail, * p;
		tail = this;
		ElemType val;
		cin >> val;
		while (val)
		{
			p = (Node*)malloc(LENG);
			p->data = val;
			tail->next = p;
			tail = p;
			data++;
			cin >> val;
		}
		tail->next = NULL;
	}
	void print()
	{
		Node* p = next;
		while (p)
		{
			cout << p->data << " ";
			p = p->next;
		}
		putchar('\n');
	}
	void UpOrderInsert(ElemType val)
	{
		Node* p = this, * q = next;
		while (q && q->data < val)
		{
			p = q;
			q = q->next;
		}
		Node* temp = (Node*)malloc(LENG);
		temp->data = val;
		temp->next = q;
		p->next = temp;
		data++;
	}
	int Insert(int i, ElemType val)
	{
		if (i < 1) return 0;
		int j = 1;
		Node* p = this;
		while (p && j < i)
		{
			p = p->next;
			j++;
		}
		if (!p) return 0;
		Node* temp = (Node*)malloc(LENG);
		temp->data = val;
		temp->next = p->next;
		p->next = temp;
		data++;
		return 1;
	}
	int DeleteByVal(ElemType val)
	{
		Node* p = this, * q = next;
		while (q && q->data != val)
		{
			p = q;
			q = q->next;
		}
		if (q)  //有元素为val的结点
		{
			p->next = q->next;
			free(q);
			data--;
			return 1;
		}
		return 0;
	}
	ElemType DeleteByPos(int pos)	//删除指点位置元素
	{
		if (pos < 1) return 0;
		Node* p = this;
		int i = 1;
		while (p->next && i < pos)
		{
			p = p->next;
			i++;
		}
		if (p->next == NULL) return 0;
		Node* q = p->next;
		p->next = q->next;
		ElemType del = q->data;
		free(q);
		data--;
		return del;
	}
	void MergeList(Node* Lb)	//合并两个有序单链表为一个
	{
		Node* pa = next, * pb = Lb->next;
		Node* tail = this;   //使用表La的头指针，tail为尾指针
		//不能free(Lb)，因为此处Lb是类定义生成而不是malloc()得到的;
		while (pa && pb)
		{
			if (pa->data <= pb->data)  //取表La的一个结点
			{
				tail->next = pa;  //插在表Lc的尾结点之后
				tail = pa;  //变为表Lc的新尾结点
				pa = pa->next;  //移向表La的下一个结点
			}
			else
			{
				tail->next = pb;
				tail = pb;
				pb = pb->next;
			}
		}
		if (pa) tail->next = pa;  //插入表La的剩余段
		else tail->next = pb;  //插入表Lb的剩余段
		data += Lb->data;
	}
	ElemType GetElem(int pos)
	{
		if (pos < 1) return 0;
		Node* p = next;
		int i = 1;
		while (p && i < pos)
		{
			p = p->next;
			i++;
		}
		if (!p) return 0;
		return p->data;
	}
	void NodeSort(void)	//实现将单链表L结点重排，使其递增有序
	{
		Node* p = next, * q = p->next, * pre;	//q保持p后继结点指针，以保证不断链
		p->next = NULL;   //构造只含一个数据结点的有序表
		p = q;
		while (p)
		{
			q = p->next;  //保存p的后继结点指针
			pre = this;
			while (pre->next != NULL && pre->next->data < p->data)
				pre = pre->next;  //在有序表中查找插入的前驱结点pre
			p->next = pre->next;  //将p插入到pre之后
			pre->next = p;
			p = q;	//扫描原单链表中剩下的结点
		}
	}
	void TriLinklistAnd(Node* B, Node* C)//A，B，C为3个递增有序的线性表，删除A中与B且C中相同的元素
	{
		Node* pa = this, * pb = B->next, * pc = C->next;
		while ((pa->next) && pb && pc)
		{
			if ((pa->next->data == pb->data) && (pa->next->data == pc->data))
			{
				pb = pb->next; pc = pc->next;
				Node* q = pa->next;
				pa->next = q->next;
				free(q);
				data--;
				if (!pb || !pc) break;
			}
			if (pa->next->data < pb->data || pa->next->data < pc->data)
				pa = pa->next;
			else if (pa->next->data > pb->data) pb = pb->next;
			else if (pa->next->data > pc->data) pc = pc->next;
		}
	}
	void RemoveElements(ElemType val)
	{
		Node* temp = this;
		while (temp->next)
		{
			if (temp->next->data == val)
			{
				Node* del = temp->next;
				temp->next = del->next;
				free(del);
				data--;
			}
			else temp = temp->next;
		}
	}
	Node* RemoveElements_RecursionHelper(Node* head, ElemType val)
	{
		if (!head) return head;
		head->next = RemoveElements_RecursionHelper(head->next, val);
		return head->data == val ? head->next : head;
	}
	void RemoveElements_Recursion(ElemType val)//warning:递归未处理data
	{
		next = RemoveElements_RecursionHelper(next, val);
	}
	void ReverseList(void)
	{
		Node* pre = NULL, * cur = next;
		while (cur)
		{
			Node* next = cur->next;
			cur->next = pre;
			pre = cur;
			cur = next;
		}
		next = pre;
	}
	Node* ReverseList_RecursionHelper(Node* head)
	{
		if (!head || !head->next)
			return head;
		Node* newHead = ReverseList_RecursionHelper(head->next);
		head->next->next = head;
		head->next = NULL;
		return newHead;
	}
	void ReverseList_Recursion(void)
	{
		next = ReverseList_RecursionHelper(next);
	}
	void SwapPairs(void)//两两交换链表节点
	{
		Node* temp = this;
		while (temp->next && temp->next->next)
		{
			Node* node1 = temp->next, * node2 = node1->next;
			temp->next = node2;
			node1->next = node2->next;
			node2->next = node1;
			temp = node1;
		}
	}
	Node* SwapPairs_RecursionHelper(Node* head)
	{
		if (!head || !head->next) return head;
		Node* newHead = head->next;
		head->next = SwapPairs_RecursionHelper(newHead->next);
		newHead->next = head;
		return newHead;
	}
	void SwapPairs_Recursion(void)
	{
		next = SwapPairs_RecursionHelper(next);
	}
	void RemoveNthFromEnd(int n)
	{
		if (n > data || n <= 0) return;
		Node* first = next, * second = this;
		for (int i = 0; i < n; i++)
			first = first->next;
		//first比second超前n个节点, first和second同时对链表进行遍历
		//当first遍历到链表的末尾时, second就恰好处于倒数第n个节点
		//second 指向哑节点时,second 的下一个节点就是我们需要删除的节点
		while (first)
		{
			first = first->next;
			second = second->next;
		}
		Node* del = second->next;
		second->next = del->next;
		free(del);
		data--;
	}
	void RemoveNthFromEnd_Stack(int n)
	{
		if (n > data || n <= 0) return;
		class Stack
		{
		private:
			Node* data;
			Stack* next;
		public:
			Stack() :data(NULL), next(NULL)
			{}
			~Stack()
			{}
			void Push(Node* t)
			{
				Stack* p = (Stack*)malloc(sizeof(Stack));
				p->data = t;
				p->next = next;
				next = p;
				data++;
			}
			Node* Pop(void)
			{
				Stack* p = next;
				if (!p)
				{
					printf("栈为空，无法出栈\n");
					return 0;
				}
				Node* t = p->data;
				next = p->next;
				free(p);
				data--;
				return t;
			}
		};
		Stack S;
		Node* cur = this;
		while (cur)
		{
			S.Push(cur);
			cur = cur->next;
		}
		for (int i = 0; i < n; ++i)
			S.Pop();
		Node* pre = S.Pop();
		Node* del = pre->next;
		pre->next = del->next;
		free(del);
		data--;
	}
	Node* GetIntersectionNode(Node* LB)
	{
		Node* pa = next, * pb = LB->next;
		//A长度为a,B长度为b,假设存在交叉点，此时A到交叉点距离为c,而B到交叉点距离为d
		//后续交叉后长度是一样的，那么就是 a - c = b - d  ->  a + d = b + c
		//这里意味着,分别让A和B额外多走一遍B和A，那么必然会走到交叉
		//边缘情况:都走到null依然没交叉，那么正好返回null即可
		while (pa != pb)
		{
			pa = pa ? pa->next : LB->next;
			pb = pb ? pb->next : next;
		}
		return pa;
		//朴素解法
		//求长度差,调整较长链指针,让pa和pb在同一起点上（末尾位置对齐）
		//遍历pa和pb,遇到相同则直接返回
	}
	Node* DetectCycle(void)//返回链表开始入环的第一个节点。如果链表无环,则返回null
	{
		//若无环，fast走到链表末端，返回NULL
		//若有环,两指针一定会相遇。因为每走1轮,fast与slow的间距+1,fast终会追上slow
		//设链表共有a+b个节点，其中链表头部到链表入口有a个节点,链表环有b个节点(a和b是未知数)
		//设两指针分别分别走了f,s步,有 f = 2s;		f = s + nb;(重合时fast比slow多走环的长度整数倍)
		//可得 s = nb
		//从head结点走到入环点需要走: a + nb,而slow已经走了nb，那么slow再走a步就是入环点了
		//如何知道slow刚好走了a步？ 从head开始，和slow指针一起走，相遇时刚好就是a步
		Node* slow = next, * fast = next;
		while (fast)
		{
			slow = slow->next;
			if (!fast->next) return NULL;
			fast = fast->next->next;
			if (fast == slow)
			{
				Node* ptr = next;
				while (ptr != slow)
				{
					ptr = ptr->next;
					slow = slow->next;
				}
				return ptr;
			}
		}
		return NULL;
	}
	//DetectCycle_HashTable()
	typedef struct HashTable
	{
		Node* key;
		UT_hash_handle hh;
	}HashTable;
	HashTable* hashtable;
	HashTable* find(Node* ikey)
	{
		HashTable* tmp;
		HASH_FIND_PTR(hashtable, &ikey, tmp);
		return tmp;
	}
	void insert(Node* ikey)
	{
		HashTable* tmp = (HashTable*)malloc(sizeof(HashTable));
		tmp->key = ikey;
		HASH_ADD_PTR(hashtable, key, tmp);
	}
	Node* DetectCycle_HashTable(Node* head)
	{
		hashtable = NULL;
		while (head)
		{
			if (find(head))
				return head;
			insert(head);
			head = head->next;
		}
		return NULL;
	}



};



//3.Stack(链栈)
typedef int ElemType;
#define LENG sizeof(Stack)  //结点所占单元数
class Stack
{
private:
	ElemType data;
	Stack* next;
public:
	Stack() :data(0), next(NULL)
	{}
	~Stack()
	{}
	void Create()
	{
		ElemType val;
		Stack* temp;
		cin >> val;
		while (val)
		{
			temp = (Stack*)malloc(LENG);
			temp->data = val;
			temp->next = next;
			next = temp;
			data++;
			cin >> val;
		}
	}
	void print()
	{
		Stack* p = next;
		while (p)
		{
			cout << p->data << " ";
			p = p->next;
		}
		putchar('\n');
	}
	void Clear()
	{
		while (data)
			Pop();
		free(next);
		next = NULL;
	}
	void Push(ElemType val)
	{
		Stack* p = (Stack*)malloc(LENG);
		p->data = val;
		p->next = next;
		next = p;
		data++;
	}
	ElemType Pop(void)
	{
		Stack* p = next;
		if (!p)
		{
			printf("栈为空，无法出栈\n");
			return 0;
		}
		ElemType t = p->data;
		next = p->next;
		free(p);
		data--;
		return t;
	}
	bool Empty(void)
	{
		return !next;
	}
	ElemType* MonoStack(ElemType* num, int len, const char* op)
	{
		Clear();
		ElemType* f = (ElemType*)malloc(len * sizeof(ElemType));
		
		if (!strcmp(op, "up"))
		{
			//定义函数f(i)代表数列中第i个元素之后第一个大于ai的元素的下标,不存在则f(i) = 0
			for (int i = len; i >= 1; i--)
			{
				while (data && num[next->data] <= num[i])
					Pop();
				if (data) f[i] = next->data;
				else f[i] = 0;
				Push(i);
			}
		}
		else
		{
			//定义函数f(i)代表数列中第i个元素之后第一个小于ai的元素的下标
			for (int i = len; i >= 1; i--)
			{
				while (data && num[next->data] >= num[i])
					Pop();
				if (data) f[i] = next->data;
				else f[i] = 0;
				Push(i);
			}
		}
		return f;
	}
	//用两个栈模拟队列
	class QueueByStack
	{
	private:
		Stack* Sin, * Sout;
	public:
		void InitStack(Stack** S)
		{
			*S = (Stack*)malloc(LENG);
			(*S)->data = 0;
			(*S)->next = NULL;
		}
		void InitQueue(void)
		{
			InitStack(&Sin);
			InitStack(&Sout);
		}
		void EnQueue(int x)
		{
			Sin->Push(x);
		}
		ElemType DeQueue(void)
		{
			if (Sout->data) return Sout->Pop();
			while (Sin->data)
				Sout->Push(Sin->Pop());
			return Sout->Pop();
		}
		ElemType Peek(void)
		{
			ElemType ans = DeQueue();
			Sout->Push(ans);
			return ans;
		}
		bool Empty(void)
		{
			return !(Sin->data || Sout->data);
		}
		void Clear(void)
		{
			while (Sin->data) Sin->Pop();
			while (Sout->data) Sout->Pop();
			free(Sin);
			free(Sout);
		}
	}QueueInStack;

};


//4.Dnode(双向循环链表)
typedef int ElemType;
#define LENG sizeof(Dnode)  //结点所占单元数
class Dnode
{
private:
	ElemType data;
	Dnode* prior, * next;
public:
	Dnode() :data(0), prior(NULL), next(NULL)
	{}
	~Dnode()
	{}
	void Create()
	{
		Dnode* tail, * p;
		tail = this;
		ElemType val;
		cin >> val;
		while (val)
		{
			p = (Dnode*)malloc(LENG);
			p->data = val;
			tail->next = p;
			p->prior = tail;
			tail = p;  //尾指针指向新结点
			cin >> val;
		}
		this->prior = tail;
		tail->next = this;
	}
	void print()
	{
		Dnode* p = next;
		while (p != this)
		{
			cout << p->data << " ";
			p = p->next;
		}
		putchar('\n');
	}
	void UpOrderInsert(ElemType val)
	{
		Dnode* p, * q = next;
		while (q != this && val > q->data)
		{
			q = q->next;
		}
		Dnode* temp = (Dnode*)malloc(LENG);
		p = q->prior;
		temp->data = val;
		temp->prior = p;
		temp->next = q;
		p->next = temp;
		q->prior = temp;
	}
	int Insert(int i, ElemType val)
	{
		if (i < 1) return 0;
		Dnode* p = this;
		int j = 1;
		while (j < i)
		{
			p = p->next;
			j++;
			if (p == this) return 0;
		}
		Dnode* temp = (Dnode*)malloc(LENG);
		temp->data = val;
		temp->next = p->next;
		p->next->prior = temp;
		p->next = temp;
		temp->prior = p;
		return 1;
	}
	int DeleteByVal(ElemType val)
	{
		Dnode* q = next;
		while (q != this && q->data != val)
		{
			q = q->next;
		}
		if (q != this)  //有元素为val的结点
		{
			Dnode* p = q->prior;
			p->next = q->next;
			q->next->prior = p;
			free(q);
			return 1;
		}
		return 0;   //没有删除结点
	}
	ElemType DeleteByPos(int pos)
	{
		if (pos < 1) return 0;
		Dnode* p = this;
		int i = 1;
		while (p->next != this && i < pos)
		{
			p = p->next;
			i++;
		}
		if (p->next == this) return 0;
		Dnode* q = p->next;
		p->next = q->next;
		q->next->prior = p;
		ElemType t = q->data;
		free(q);
		return t;
	}
	int Length()  //求循环链表的长度
	{
		int len = 0;
		Dnode* p = next;
		while (p != this)
		{
			len++;
			p = p->next;
		}
		return len;
	}
	void MergeList(Dnode* Lb)
	{
		Lb->prior->next = this;
		this->prior->next = Lb->next;
	}
	ElemType GetElem(int pos)
	{
		if (pos < 1) return 0;
		Dnode* p = next;
		int i = 1;
		while (p != this && i < pos)
		{
			p = p->next;
			i++;
		}
		if (p == this) return 0;
		return p->data;
	}
	void Adjust()  //L=(a1，a2，…，an)改为L=(a1，a3，…，an，…，a4，a2)，时间复杂度O(n)
	{
		Dnode* p = next, * q = this, * s, * r = p;
		int i = 0;
		while (p->next != this)
		{
			s = p->next;   //s保存p的后继
			i++;    //i值表示当前p指针的指向
			if (i % 2 == 0) //将第偶数个指针移到头指针左边
			{
				q->prior = p;
				p->next = q;
				q = q->prior;
				p = s;	//p后移
				r->next = s;   //将s指针指向的节点与修改后的前一节点相连接
				s->prior = r;
				r = r->next;
			}
			else p = p->next;   //不符合条件，p指针往后移
		}
		p->next = q;
		q->prior = p;
	}
};


//5.Date(日期类)
class Date
{
private:
	int year, month, day;
public:
	Date():year(0),month(0),day(0)
	{}
	Date(int y, int m, int d) :year(y), month(m), day(d)
	{}
	~Date()
	{}
	void SetDay(int y, int m, int d)
	{
		year = y;
		month = m;
		day = d;
	}
	bool IsLeap()   //判断是否为闰年
	{
		bool leap = false;
		if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
			leap = true;
		return leap;
	}
	int NumOfDays() //统计总天数
	{
		int days = 0;
		int daysPerMonth[13] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
		if (IsLeap()) daysPerMonth[2] = 29;
		for (int i = 1; i < month; i++)
			days += daysPerMonth[i];
		days += day;
		return days;
	}
};


//6.TSMatrix(三元组表，用于存储稀疏矩阵)
#define MAXSIZE 100
typedef int ElemType;
ElemType matrix[MAXSIZE + 1][MAXSIZE + 1];
typedef struct
{
	int i, j;   //非零元行、列下标
	ElemType e;
}Triple;   //定义三元组
class TSMatrix
{
private:
	Triple data[MAXSIZE + 1];
	int mu, nu, tu;   //mu行数，nu列数，tu非零元个数
public:
	TSMatrix() :mu(0), nu(0), tu(0), data()
	{}
	~TSMatrix()
	{}
	void InputTSMatrix()   //输入三元组表
	{
		printf("请输入行数、列数和非零元个数：\n");
		scanf("%d%d%d", &mu, &nu, &tu);
		printf("请依次输入非零元的行、列、值：\n");
		for (int k = 1; k <= tu; k++)
		{
			scanf("%d%d", &data[k].i, &data[k].j);
			cin >> data[k].e;
		}
	}
	void InputMatrix(int m, int n)
	{
		printf("请输入%d行%d列矩阵：\n", m, n);
		for (int i = 1; i <= m; i++)
			for (int j = 1; j <= n; j++)
				cin >> matrix[i][j];
	}
	void ShowTSMatrix()
	{
		printf("行 列 值\n");
		for (int i = 1; i <= tu; i++)
		{
			printf("%d  %d  ", data[i].i, data[i].j);
			cout << data[i].e << endl;
		}
	}
	void TSMatrixToMatrix()   //三元组表转化为矩阵
	{
		memset(matrix, 0, sizeof(matrix));
		for (int i = 1; i <= tu; i++)
			matrix[data[i].i][data[i].j] = data[i].e;
	}
	void ShowMatrix()   //打印矩阵
	{
		for (int i = 1; i <= mu; i++)
		{
			for (int j = 1; j <= nu; j++)
				printf("%d ", matrix[i][j]);
			putchar('\n');
		}
	}
	void MatrixToTSMatrix(int m, int n)   //矩阵转化为三元组表
	{
		tu = 0;
		mu = m; nu = n;
		for (int i = 1; i <= m; i++)
			for (int j = 1; j <= n; j++)
				if (matrix[i][j])
				{
					++tu;
					data[tu].i = i;
					data[tu].j = j;
					data[tu].e = matrix[i][j];
				}
	}
	void MatrixAdd(TSMatrix A, TSMatrix B)   //矩阵加法
	{
		Triple temp;
		mu = A.mu;
		nu = A.nu;
		tu = 0;
		int pa = 1, pb = 1;   //指针作用，分别指向A,B当前位置
		while (pa <= A.tu && pb <= B.tu)
		{
			if (A.data[pa].i == B.data[pb].i && A.data[pa].j == B.data[pb].j)
			{
				temp.e = A.data[pa].e + B.data[pb].e;
				if (temp.e)
				{
					temp.i = A.data[pa].i;
					temp.j = A.data[pa].j;
					data[++tu] = temp;
				}
				pa++; pb++;
			}
			else if (A.data[pa].i == B.data[pb].i)
			{
				if (A.data[pa].j < B.data[pb].j)
					data[++tu] = A.data[pa++];
				else data[++tu] = B.data[pb++];
			}
			else if (A.data[pa].i < B.data[pb].i)
				data[++tu] = A.data[pa++];
			else data[++tu] = B.data[pb++];
		}
		if (pa > A.tu)
		{
			while (pb <= B.tu)
				data[++tu] = B.data[pb++];
		}
		else data[++tu] = A.data[pa++];
	}
	void MatrixReverse(TSMatrix M)   //实现稀疏矩阵三元组逆置
	{
		mu = M.nu; nu = M.mu; tu = M.tu;
		if (tu)
		{
			int q, col;   //q指向T写时的位置,col->column表示第几列
			int* num = (int*)malloc((M.nu + 1) * sizeof(int));
			int* cpot = (int*)malloc((M.nu + 1) * sizeof(int));  //表示数据放在T的位置
			cpot[1] = 1;
			for (int i = 1; i <= M.nu; i++)
				num[i] = 0;
			for (int j = 1; j <= M.tu; j++)
				num[M.data[j].j]++;
			for (col = 2; col <= M.nu; col++)
				cpot[col] = cpot[col - 1] + num[col - 1];
			for (int p = 1; p <= M.tu; p++)   //扫描M三元表
			{
				col = M.data[p].j;   //确定当前元素列号
				q = cpot[col];   //确定当前元素M.data[p]在T的当前存放位置
				data[q].j = M.data[p].i;
				data[q].i = M.data[p].j;
				data[q].e = M.data[p].e;
				++cpot[col];   //T的当前列指示下一空位置
			}
		}

	}
};


//7.BiTree(二叉树)
typedef char ElemType;
#define LENG sizeof(TreeNode)
#define MaxSize 100
typedef struct TreeNode
{
	ElemType val;
	TreeNode* left;
	TreeNode* right;
}TreeNode;
//如果需要遍历整颗树，递归函数就不能有返回值。如果需要遍历某一条固定路线，递归函数就一定要有返回值！
//1.如果需要搜索整颗二叉树且不用处理递归返回值，递归函数就不要返回值。
//2.如果需要搜索整颗二叉树且需要处理递归返回值，递归函数就需要返回值。
//3.如果要搜索其中一条符合条件的路径，那么递归一定需要返回值，因为遇到符合条件的路径了就要及时返回。
class BiTree
{
private:
	void Visit(TreeNode* p)
	{
		cout << p->val << " ";
	}
public:
	TreeNode* root;
	BiTree() :root(NULL)
	{}
	~BiTree()
	{}
	void Create(TreeNode* (&root))   //T是指向根指针的指针
	{
		//按先序次序输入二叉树中结点的值(一个字符)，空格字符表示空树
		char ch;
		scanf("%c", &ch);
		if (ch == ' ') root = NULL;
		else
		{
			root = (TreeNode*)malloc(LENG);
			root->val = ch;
			Create(root->left);
			Create(root->right);
		}
	}
	struct TreeNode* BuildTreeTraverse(int* inorder, int inorderBegin, int inorderEnd, int* postorder, int postorderBegin, int postorderEnd)
	{
		// 在切割的过程中会产生四个区间，把握不好不变量的话，一会左闭右开，一会左闭又闭，必然乱套！
		// 中序区间：[inorderBegin, inorderEnd)，后序区间[postorderBegin, postorderEnd)
		// 如果数组大小为零的话，说明是空节点了。
		if (postorderBegin == postorderEnd) return NULL;
		// 如果不为空，那么取后序数组最后一个元素作为节点元素。
		int rootValue = postorder[postorderEnd - 1];
		struct TreeNode* root = (struct TreeNode*)calloc(1, sizeof(struct TreeNode));
		root->val = rootValue;
		root->left = NULL;
		root->right = NULL;
		// 叶子节点
		if (postorderEnd - postorderBegin == 1) return root;
		// 找到后序数组最后一个元素在中序数组的位置，作为切割点
		int delimiterIndex;
		for (delimiterIndex = inorderBegin; delimiterIndex < inorderEnd; ++delimiterIndex)
			if (inorder[delimiterIndex] == rootValue) break;
		// 坚持左闭右开的原则
		// 左闭右开区间：[0, delimiterIndex) && [delimiterIndex + 1, end)
		// 切割中序数组，切成中序左数组和中序右数组 （顺序别搞反了，一定是先切中序数组）
		// 左中序区间，左闭右开[leftInorderBegin, leftInorderEnd)
		int leftInorderBegin = inorderBegin;
		int leftInorderEnd = delimiterIndex;
		// 右中序区间，左闭右开[rightInorderBegin, rightInorderEnd)
		int rightInorderBegin = delimiterIndex + 1;
		int rightInorderEnd = inorderEnd;
		// 切割后序数组，切成后序左数组和后序右数组
		// 后序数组没有明确的切割元素来进行左右切割，不像中序数组有明确的切割点
		// 此时有一个很重要的点，就是中序数组大小一定是和后序数组的大小相同的（这是必然）
		// 中序数组我们都切成了左中序数组和右中序数组了，那么后序数组就可以按照左中序数组的大小来切割，切成左后序数组和右后序数组
		// 左后序区间，左闭右开[leftPostorderBegin, leftPostorderEnd),注意这里使用了左中序数组大小作为切割点：[0, leftInorder.size)
		int leftPostorderBegin = postorderBegin;
		int leftPostorderEnd = postorderBegin + delimiterIndex - inorderBegin;// 终止位置是 需要加上 中序区间的大小size
		// 右后序区间，左闭右开[rightPostorderBegin, rightPostorderEnd)
		int rightPostorderBegin = postorderBegin + (delimiterIndex - inorderBegin);
		int rightPostorderEnd = postorderEnd - 1;// 后序数组的最后一个元素指定不能要了，这是切割点 也是 当前二叉树中间节点的元素，已经用了
		// 递归处理左区间和右区间
		root->left = BuildTreeTraverse(inorder, leftInorderBegin, leftInorderEnd, postorder, leftPostorderBegin, leftPostorderEnd);
		root->right = BuildTreeTraverse(inorder, rightInorderBegin, rightInorderEnd, postorder, rightPostorderBegin, rightPostorderEnd);
		return root;
	}
	struct TreeNode* BuildTreeWithInAndPost(int* inorder, int inorderSize, int* postorder, int postorderSize) {
		if (!inorderSize || !postorderSize) return NULL;
		// 左闭右开的原则
		return BuildTreeTraverse(inorder, 0, inorderSize, postorder, 0, postorderSize);
	}


	void PreOrderTraverse_Recursion(TreeNode* root)
	{
		if (!root) return;
		Visit(root);
		PreOrderTraverse_Recursion(root->left);
		PreOrderTraverse_Recursion(root->right);
	}
	void PreOrderTraverse(TreeNode* root)
	{
		if (!root) return;
		TreeNode* Stack[MaxSize], * p;	//定义一个栈
		int top = -1;	//初始化栈
		Stack[++top] = root;	//根节点入栈
		while (top != -1)	//栈空循环退出，遍历结束
		{
			p = Stack[top--];	//出栈并输出栈顶结点
			Visit(p);
			if (p->right)	//栈顶结点的右孩子存在，则右孩子入栈
				Stack[++top] = p->right;
			if (p->left)	//栈顶结点的左孩子存在，则右孩子入栈
				Stack[++top] = p->left;
		}
		putchar('\n');
	}
	void InOrderTraverse_Recursion(TreeNode* root)
	{
		if (!root) return;
		InOrderTraverse_Recursion(root->left);
		Visit(root);
		InOrderTraverse_Recursion(root->right);
	}
	void InOrderTraverse(TreeNode* root)
	{
		if (!root) return;
		TreeNode* Stack[MaxSize], * p = root;	//定义一个栈
		int top = -1;
		//下面这个循环完成中序遍历，注意：进栈、出栈过程可能出现栈空状态
		//但此时遍历还未结束，因根节点的右子树还未遍历，此时p非空，根据这一点维持循环的进行
		while (top != -1 || p)
		{
			while (p)	//左子树存在，进栈
			{
				Stack[++top] = p;
				p = p->left;
			}
			if (top != -1)	//在栈未空的情况下，出栈并输出出栈结点
			{
				p = Stack[top--];
				Visit(p);
				p = p->right;
			}
		}
		putchar('\n');
	}
	void PostOrderTraverse_Recursion(TreeNode* root)
	{
		if (!root) return;
		PostOrderTraverse_Recursion(root->left);
		PostOrderTraverse_Recursion(root->right);
		Visit(root);
	}
	void PostOrderTraverse(TreeNode* root)
	{
		if (!root) return;
		TreeNode* Stack1[MaxSize], * Stack2[MaxSize], * p = NULL;	//定义两个栈
		int top1 = -1, top2 = -1;
		Stack1[++top1] = root;
		while (top1 != -1)
		{
			p = Stack1[top1--];
			Stack2[++top2] = p;		//注意这里和先序遍历的区别，输出改为入栈Stack2
			//注意下边这两个if语句和先序遍历的区别，左、右子树的入栈顺序相反
			if (p->left)
				Stack1[++top1] = p->left;
			if (p->right)
				Stack1[++top1] = p->right;
		}
		while (top2 != -1)
		{
			//出栈序列即为后序遍历序列
			p = Stack2[top2--];
			Visit(p);
		}
		putchar('\n');
	}
	int* DFSTraverse_Unified_Pre(TreeNode* root, int* returnSize) 
	{
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* Stack[MaxSize], *p;
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			p = Stack[--top];
			if (p)
			{
				//右
				if (p->right)   
				{
					Stack[top++] = p->right;
				}
				//左
				if (p->left)    
				{
					Stack[top++] = p->left;
				}
				//中
				Stack[top++] = p;  
				Stack[top++] = NULL;
			}
			else
			{
				p = Stack[--top];
				ans[(*returnSize)++] = p->val;
			}
		}
		return ans;
	}
	int* DFSTraverse_Unified_In(TreeNode* root, int* returnSize)
	{
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* Stack[MaxSize], * p;
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			p = Stack[--top];
			if (p)
			{
				//右
				if (p->right)
				{
					Stack[top++] = p->right;
				}
				//中
				Stack[top++] = p;
				Stack[top++] = NULL;
				//左
				if (p->left)
				{
					Stack[top++] = p->left;
				}				
			}
			else
			{
				p = Stack[--top];
				ans[(*returnSize)++] = p->val;
			}
		}
		return ans;
	}
	int* DFSTraverse_Unified_Post(TreeNode* root, int* returnSize) 
	{
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* Stack[MaxSize], *p;
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			p = Stack[--top];
			if (p)
			{
				//中
				Stack[top++] = p;
				Stack[top++] = NULL;
				//右
				if (p->right)
				{
					Stack[top++] = p->right;
				}
				//左
				if (p->left)
				{
					Stack[top++] = p->left;
				}

			}
			else
			{
				p = Stack[--top];
				ans[(*returnSize)++] = p->val;
			}
		}
		return ans;
	}
	int* DFSTraverse_Morris_Pre(TreeNode* root, int* returnSize)
	{
		//特殊处理：
		//1.建立连接的同时输出此根结点。
		//2.到达一些没有子节点的叶子节点，直接输出并向右走返回上层或向此节点的右子树前进。
		//3.判断出某节点已有连接，则不用输出，直接断开走过的连接后继续向右走。
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* curPtr = root, * MostRightPtr;
		while (curPtr)
		{
			MostRightPtr = curPtr->left;
			if (MostRightPtr)   // 当前结点的左子树存在即可建立连接
			{
				// 找到当前左子树的最右侧节点，并且不能沿着连接返回上层
				while (MostRightPtr->right && MostRightPtr->right != curPtr)
					MostRightPtr = MostRightPtr->right;
				//最右侧节点的右指针没有指向根结点，创建连接并往下一个左子树的根结点进行连接操作
				if (!MostRightPtr->right)
				{
					MostRightPtr->right = curPtr;
					ans[(*returnSize)++] = curPtr->val;
					curPtr = curPtr->left;
					continue;   // 这个continue很关键
				}
				// 当左子树的最右侧节点有指向根结点
				// 此时说明我们已经进入到了返回上层的阶段，不再是一开始的建立连接阶段，
				// 同时在回到根结点时我们应已输出过下层节点，直接断开连接即可
				else
					MostRightPtr->right = NULL;

			}
			//当前节点的左子树为空，说明左侧到头，直接输出
			else
				ans[(*returnSize)++] = curPtr->val;
			// 返回上层的阶段不断向右走
			curPtr = curPtr->right;
		}
		return ans;
	}
	int* DFSTraverse_Morris_In(TreeNode* root, int* returnSize)
	{
		//特殊处理：
		//1.在建立连接阶段并不输出结点。
		//2.在找到最左侧结点（即根结点的左子树为空）时，开始向右走返回上层并同时输出当前结点。
		//3.对右子树也进行同样的处理。
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		struct TreeNode* curPtr = root, * MostRightPtr;
		while (curPtr)
		{
			MostRightPtr = curPtr->left;
			if (MostRightPtr)   // 当前结点的左子树存在即可建立连接
			{
				// 找到当前左子树的最右侧节点，并且不能沿着连接返回上层
				while (MostRightPtr->right && MostRightPtr->right != curPtr)
					MostRightPtr = MostRightPtr->right;
				//最右侧节点的右指针没有指向根结点，创建连接并往下一个左子树的根结点进行连接操作
				if (!MostRightPtr->right)
				{
					MostRightPtr->right = curPtr;
					curPtr = curPtr->left;
					continue;   // 这个continue很关键
				}				
				//当左子树的最右侧节点有指向根结点
				//此时说明我们已经进入到了返回上层的阶段，不再是一开始的建立连接阶段
				//同时在回到根结点时我们应已输出过下层节点，直接断开连接即可
				else
					MostRightPtr->right = NULL;

			}
			//当前节点的左子树为空，说明左侧到头，直接输出并返回上层
			ans[(*returnSize)++] = curPtr->val;
			// 返回上层的阶段不断向右走
			curPtr = curPtr->right;
		}
		return ans;
	}	
	struct TreeNode* postMorrisReverseList(struct TreeNode* head)//翻转单链表函数
	{
		struct TreeNode* cur = head;
		struct TreeNode* pre = NULL;    //哨兵结点
		while (cur)
		{
			struct TreeNode* next = cur->right;
			cur->right = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}	
	void postMorrisPrint(struct TreeNode* head, int* ans, int* returnSize)//输出函数
	{
		struct TreeNode* newhead = postMorrisReverseList(head);     //newhead为翻转后的新头部
		struct TreeNode* cur = newhead;
		while (cur)
		{
			ans[(*returnSize)++] = cur->val;
			cur = cur->right;
		}
		postMorrisReverseList(newhead); //遍历结束后再次翻转恢复原链表
	}
	int* DFSTraverse_Morris_Post(TreeNode* root, int* returnSize)
	{
		//特殊处理：
		//1.我们将一个节点的连续右节点当成一个单链表来看待，可以发现，输出顺序是将此单链表翻转后输出。
		//2.当我们返回上层之后，也就是将连线断开的时候,输出下层的单链表,只需要将这个单链表逆序输出即可
		//3.不应该打印当前层，而是之前的一层，否则根结点会先与右边输出。
		*returnSize = 0;
		int* ans = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		struct TreeNode* curPtr = root, * MostRightPtr;
		while (curPtr)
		{
			MostRightPtr = curPtr->left;
			if (MostRightPtr)   // 当前结点的左子树存在即可建立连接
			{
				// 找到当前左子树的最右侧节点，并且不能沿着连接返回上层
				while (MostRightPtr->right && MostRightPtr->right != curPtr)
					MostRightPtr = MostRightPtr->right;
				//最右侧节点的右指针没有指向根结点，创建连接并往下一个左子树的根结点进行连接操作
				if (!MostRightPtr->right)
				{
					MostRightPtr->right = curPtr;
					curPtr = curPtr->left;
					continue;   // 这个continue很关键
				}
				//当左子树的最右侧节点有指向根结点
				//此时说明我们已经进入到了返回上层的阶段，不再是一开始的建立连接阶段
				//断开连接同时对之前的一层进行翻转并输出
				else
				{
					MostRightPtr->right = NULL;
					postMorrisPrint(curPtr->left, ans, returnSize);
				}

			}
			// 返回上层的阶段不断向右走
			curPtr = curPtr->right;
		}
		// 最后一轮循环结束时，从root结点引申的右结点单链表并没有输出，这里补上
		postMorrisPrint(root, ans, returnSize);
		return ans;
	}
	void LevelOrderTraverse(TreeNode* root)
	{
		if (!root) return;
		int front = 0, rear = 0;
		TreeNode* queue[MaxSize];	//定义一个循环队列，用来记录将要访问的层次上的结点
		TreeNode* q;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;	//根节点入队
		while (front != rear)	//当队列不为空时进入循环
		{
			front = (front + 1) % MaxSize;
			q = queue[front];	//队首结点出队
			Visit(q);	//访问队首结点
			if (q->left)	//左子树根节点入队
			{
				rear = (rear + 1) % MaxSize;
				queue[rear] = q->left;
			}
			if (q->right)	//右子树根节点入队
			{
				rear = (rear + 1) % MaxSize;
				queue[rear] = q->right;
			}
		}
		putchar('\n');
	}
	int** LevelOrderTraverse_Perlevel(TreeNode* root, int* returnSize, int** returnColumnSizes) 
	{
		*returnSize = 0;
		int** ans = (int**)calloc(MaxSize, sizeof(int*));
		*returnColumnSizes = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* queue[MaxSize], q;
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int level = 0, cnt = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			(*returnColumnSizes)[level] = cnt;
			ans[level] = (int*)calloc(cnt, sizeof(int));
			for (int i = 0; i < cnt; i++)
			{
				front = (front + 1) % MaxSize;
				TreeNode* p = queue[front];
				ans[level][i] = p->val;
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
			}
			level++;
		}
		*returnSize = level;
		return ans;
	}
	int** LevelOrderTraverse_Perlevel_fromBottom(TreeNode* root, int* returnSize, int** returnColumnSizes) {
		*returnSize = 0;
		int** ans = (int**)calloc(MaxSize, sizeof(int*));
		*returnColumnSizes = (int*)calloc(MaxSize, sizeof(int));
		if (!root) return ans;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int cnt = 0, level = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			(*returnColumnSizes)[level] = cnt;
			ans[level] = (int*)calloc(cnt, sizeof(int));
			for (int i = 0; i < cnt; ++i)
			{
				front = (front + 1) % MaxSize;
				TreeNode* p = queue[front];
				ans[level][i] = p->val;
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
			}
			level++;
		}
		*returnSize = level;
		for (int i = 0; 2 * i < *returnSize; ++i)
		{
			int* tmp1 = ans[i];
			ans[i] = ans[(*returnSize) - 1 - i];
			ans[(*returnSize) - 1 - i] = tmp1;
			int tmp2 = (*returnColumnSizes)[i];
			(*returnColumnSizes)[i] = (*returnColumnSizes)[(*returnSize) - 1 - i];
			(*returnColumnSizes)[(*returnSize) - 1 - i] = tmp2;
		}
		return ans;
	}
	int GetDepth(TreeNode* root)
	{
		int ld, rd;
		if (root == NULL)
			return 0;   //空树则深度为0
		ld = GetDepth(root->left);   //求左子树深度
		rd = GetDepth(root->right);   //求右子树深度
		return (ld > rd ? ld : rd) + 1;   //返回左、右子树深度的最大值加1，即为整棵树深度
	}
	int GetDepth_Iteration(TreeNode* root)
	{
		if (!root) return 0;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int level = 0, cnt = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			for (int i = 0; i < cnt; ++i)
			{
				front = (front + 1) % MaxSize;
				TreeNode* p = queue[front];
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
			}
			++level;
		}
		return level;
	}
	void GetDepth_Backtracking_Helper(struct TreeNode* node, int depth, int* result)
	{
		*result = depth > *result ? depth : *result; // 中
		if (node->left == NULL && node->right == NULL) return;
		if (node->left) 
		{ // 左
			depth++;    // 深度+1
			GetDepth_Backtracking_Helper(node->left, depth, result);
			depth--;    // 回溯，深度-1
		}
		if (node->right) 
		{ // 右
			depth++;    // 深度+1
			GetDepth_Backtracking_Helper(node->right, depth, result);
			depth--;    // 回溯，深度-1
		}
		return;
	}
	void GetDepth_Backtracking_Helper_Simplified(struct TreeNode* node, int depth, int* result)
	{
		*result = depth > *result ? depth : *result; // 中
		if (!node->left && !node->right) return;
		if (node->left)	//左,回溯体现在括号中+1
			GetDepth_Backtracking_Helper_Simplified(node->left, depth + 1, result);
		if (node->right)
			GetDepth_Backtracking_Helper_Simplified(node->right, depth + 1, result);
		return;
	}
	int GetDepth_Backtracking(TreeNode* root)
	{
		int result = 0;
		if (!root) return result;
		GetDepth_Backtracking_Helper(root, 1, &result);
		// OR  GetDepth_Backtracking_Helper_Simplified(root, 1, &result);
		return result;
	}

	int MinDepth(TreeNode* root)
	{
		if (!root) return 0;
		int lmin = MinDepth(root->left);
		int rmin = MinDepth(root->right);
		//1.如果左孩子和右孩子有为空的情况，直接返回lmin+rmin+1
		//如果左孩子和右孩子其中一个为空，那么需要返回比较大的那个孩子的深度 
		//这里其中一个节点为空，说明lmin和rmin有一个必然为0，所以可以返回lmin + rmin + 1;
		//2.如果都不为空，返回较小深度+1
		return !root->left || !root->right ? lmin + rmin + 1 : min(lmin, rmin) + 1;
	}
	int MinDepth_BFS(TreeNode* root) {
		if (!root) return 0;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int cnt = 0, level = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			for (int i = 0; i < cnt; ++i)
			{
				front = (front + 1) % MaxSize;
				TreeNode* p = queue[front];
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
				if (!p->left && !p->right)
				{
					return level + 1;   //level + 1即当前depth
				}
			}
			++level;
		}
		return level;
	}
	bool IsBalanced_FromTopDown(TreeNode* root) 
	{
		//O(n^2),O(n),函数 GetDepth() 会被重复调用，导致时间复杂度较高
		if (!root) return true;
		return abs(GetDepth(root->left) - GetDepth(root->right)) <= 1 && IsBalanced_FromTopDown(root->left) && IsBalanced_FromTopDown(root->right);
	}
	int BalanceHeight(struct TreeNode* root) 
	{
		if (root == NULL)
			return 0;
		int leftDepth = BalanceHeight(root->left);
		if (leftDepth == -1) return -1; // 说明左子树已经不是二叉平衡树
		int rightDepth = BalanceHeight(root->right);
		if (rightDepth == -1) return -1; // 说明右子树已经不是二叉平衡树
		return abs(leftDepth - rightDepth) > 1 ? -1 : 1 + max(leftDepth, rightDepth);
	}
	bool IsBalanced_FromBottomUp(struct TreeNode* root) 
	{
		//O(n),O(n) 自底向上的做法，则对于每个节点，函数 BalanceHeight() 只会被调用一次
		return BalanceHeight(root) == -1 ? false : true;
	}
	bool IsBalanced_Iteration(TreeNode* root)
	{
		if (!root) return true;
		struct TreeNode* Stack[MaxSize];
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			struct TreeNode* node = Stack[--top];
			if (abs(GetDepth_Iteration(node->left) - GetDepth_Iteration(node->right)) > 1)
				return false;
			if (node->right) Stack[top++] = node->right;
			if (node->left) Stack[top++] = node->left;
		}
		return true;
	}
	int FindBottomLeftValue(struct TreeNode* root) 
	{
		//改进方案,入队时从右到左入,则最后一个元素即是所求叶子结点，直接return p->val;	省略中间判断
		struct TreeNode* queue[MaxSize], q;
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int cnt = 0;
		int ans = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			for (int i = 0; i < cnt; i++)
			{
				front = (front + 1) % MaxSize;
				struct TreeNode* p = queue[front];
				if (i == 0) ans = p->val;   // 记录每一行第一个元素，循环结束后即为左下角元素
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
			}
		}
		return ans;
	}
	void RenewBottomLeftValue(struct TreeNode* root, int leftLen, int* maxLen, int* mostLeftVal)
	{
		//如何判断是最后一行呢，其实就是深度最大的叶子节点一定是最后一行
		//如果找最左边的呢？可以使用前序遍历，这样才先优先左边搜索
		//然后记录深度最大的叶子节点，此时就是树的最后一行最左边的值
		if (!root->left && !root->right)
		{
			if (leftLen > *maxLen)
			{
				*maxLen = leftLen;	// 更新最大深度
				*mostLeftVal = root->val;	// 最大深度最左面的数值
			}
			return;
		}	// 中
		if (root->left)	// 左
			RenewBottomLeftValue(root->left, leftLen + 1, maxLen, mostLeftVal);// 隐藏着回溯
		if (root->right)	// 右
			RenewBottomLeftValue(root->right, leftLen + 1, maxLen, mostLeftVal);
		//等价于
		//leftLen++;	// 深度加一
		//RenewBottomLeftValue(root->right, leftLen, maxLen, mostLeftVal);
		//leftLen--;  // 回溯，深度减一
	}
	int FindBottomLeftValue_Recursion(struct TreeNode* root) 
	{
		int maxLen = INT_MIN, mostLeftVal = root->val;//记录最大深度,记录最大深度最左节点的数值
		RenewBottomLeftValue(root, 0, &maxLen, &mostLeftVal);
		return mostLeftVal;
	}


	void InvertBiTree_Recursion(TreeNode* root)
	{
		if (!root) return;
		TreeNode* temp = root->right;
		root->right = root->left;
		root->left = temp;
		InvertBiTree_Recursion(root->left);		//递归交换左子树
		InvertBiTree_Recursion(root->right);		//递归交换右子树
		//函数返回时就表明当前结点及其左右子树均翻转了
	}
	void InvertBiTree_Recursion_Inorder(TreeNode* root)
	{
		if (!root) return;
		InvertBiTree_Recursion(root->left);		//递归交换左子树
		TreeNode* temp = root->right;
		root->right = root->left;
		root->left = temp;
		//注意 这里依然要遍历左孩子，因为中间节点已经翻转了
		InvertBiTree_Recursion(root->left);		//递归交换右子树
		//函数返回时就表明当前结点及其左右子树均翻转了
	}
	void InvertBiTree_DFS(TreeNode* root)
	{
		if (!root) return;
		TreeNode* stack[MaxSize];
		int top = 0;
		stack[top++] = root;
		while (top)
		{
			TreeNode* p = stack[--top];
			TreeNode* temp = p->left;
			p->left = p->right;
			p->right = temp;
			if (p->right) stack[top++] = p->right;
			if (p->left) stack[top++] = p->left;
		}
	}
	void InvertBiTree_UnifiedDFS(TreeNode* root)
	{
		//此处为先序,中序和后序同理,小幅修改即可
		if (!root) return;
		TreeNode* Stack[MaxSize], * p;
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			p = Stack[--top];
			if (p)
			{
				//右
				if (p->right)
				{
					Stack[top++] = p->right;
				}
				//左
				if (p->left)
				{
					Stack[top++] = p->left;
				}
				//中
				Stack[top++] = p;
				Stack[top++] = NULL;
			}
			else
			{
				p = Stack[--top];
				TreeNode* temp = root->right;
				root->right = root->left;
				root->left = temp;
			}
		}
	}
	void InvertBiTree_BFS(TreeNode* root)
	{
		if (!root) return;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root;
		int cnt = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			for (int i = 0; i < cnt; ++i)
			{
				front = (front + 1) % MaxSize;
				TreeNode* p = queue[front];
				TreeNode* temp = p->left;
				p->left = p->right;
				p->right = temp;
				if (p->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->left;
				}
				if (p->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear] = p->right;
				}
			}
		}
	}
	int GetWidth(TreeNode* T)
	{
		if (!T) return 0;	//空树直接返回0
		typedef struct
		{
			TreeNode* p;		//结点指针
			int level;		//结点所在层次号
		}St;	//非循环队列的队列元素结构体构造
		St queue[MaxSize];
		int front = 0, rear = 0;	//定义非循环队列
		int Lno, n, max = 0;
		TreeNode* q;
		++rear;
		queue[rear].p = T;	//根节点入队
		queue[rear].level = 1;	//根节点所在层次号设置为1，此为已知条件
		while (front != rear)
		{
			++front;
			q = queue[front].p;
			Lno = queue[front].level;	//Lno用来存取当前结点的层次号
			if (q->left)
			{
				++rear;
				queue[rear].p = q->left;
				queue[rear].level = Lno + 1;	//根据当前结点的层次号推知其子结点的层次号
			}
			if (q->right)
			{
				++rear;
				queue[rear].p = q->right;
				queue[rear].level = Lno + 1;	//根据当前结点的层次号推知其子结点的层次号
			}
		}//循环结束时，Lno中保存的是这棵二叉树的最大层数
		for (int i = 1; i <= Lno; i++)
		{
			n = 0;
			for (int j = 1; j <= rear; j++)
				if (queue[j].level == i)
					n++;
			if (max < n) max = n;
		}
		return max;
	}
	int CompleteBiTree_countNodes(struct TreeNode* root) 
	{
		//完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。
		//对于情况一，可以直接用 2^(树深度 - 1) 来计算，注意这里根节点深度为1。
		//对于情况二，分别递归左孩子和右孩子，递归到某一深度一定会有左孩子或者右孩子为满二叉树
		//然后依然可以按照情况1来计算。
		if (!root) return 0;
		struct TreeNode* lptr = root->left;
		struct TreeNode* rptr = root->right;
		int leftHeight = 0, rightHeight = 0;    //这里初始为0是有目的的，为了下面求指数方便
		while (lptr)    //求左子树深度
		{
			lptr = lptr->left;
			leftHeight++;
		}
		while (rptr)    //求右子树深度
		{
			rptr = rptr->right;
			rightHeight++;
		}
		if (leftHeight == rightHeight)
			return (2 << leftHeight) - 1;   // 注意(2<<1) 相当于2^2，所以leftHeight初始为0
		return CompleteBiTree_countNodes(root->left) + CompleteBiTree_countNodes(root->right) + 1;
	}
	bool Exists(struct TreeNode* root, int level, int k)
	{
		int bits = 1 << (level - 1);
		//第h层最左侧节点一定是第2^h个节点（数值2^h的二进制表示中第h位为1）
		struct TreeNode* node = root;
		//每一步的选择都会决定时候是否需要在原来的基础上增加一个偏移量，而这个偏移量刚好是其子树末层节点个数
		//因为其必是2的整次幂，所以刚好对应一个二进制位
		while (node && bits > 0)
		{
			if (bits & k)   //k的该二进制位为1，说明在右子树
				node = node->right;
			else node = node->left;     //为0，则在左子树
			bits >>= 1;
		}
		return node;
	}
	int CompleteBiTree_countNodes_BitwiseOperation(struct TreeNode* root)
	{
		if (!root) return 0;
		//约定根节点所在层级为第0层
		int level = 0;
		struct TreeNode* node = root;
		while (node->left)
		{
			++level;
			node = node->left;
		}
		int low = 1 << level, high = (1 << (level + 1)) - 1;
		//对于最大层数为h的完全二叉树，节点个数一定在 [2^h, 2^(h+1) - 1]范围内
		//可以在该范围内通过二分查找的方式得到完全二叉树的节点个数
		//根据节点个数范围的上下界得到当前需要判断的节点个数 k
		//如果第 k 个节点存在，则节点个数一定大于或等于 k
		//如果第 k 个节点不存在，则节点个数一定小于 k
		while (low < high)
		{
			int mid = (high - low + 1 >> 1) + low;
			if (Exists(root, level, mid))
				low = mid;
			else high = mid - 1;
		}
		return low;
	}
	void ConstructPaths(struct TreeNode* root, char** paths, int* returnSize, int* sta, int top)
	{
		if (root)
		{
			if (!root->left && !root->right)    //当前结点是叶子结点
			{
				char* tmp = (char*)calloc(MaxSize, sizeof(char));
				int len = 0;
				for (int i = 0; i < top; ++i)
					len += sprintf(tmp + len, "%d->", sta[i]);
				sprintf(tmp + len, "%d", root->val);
				paths[(*returnSize)++] = tmp;   //将路径加入到答案中
			}
			else
			{
				sta[top++] = root->val; //当前结点不是叶子结点，继续递归遍历
				ConstructPaths(root->left, paths, returnSize, sta, top);// 回溯
				ConstructPaths(root->right, paths, returnSize, sta, top);
			}
		}
	}
	char** BinaryTreePaths_DFS(struct TreeNode* root, int* returnSize) {
		char** paths = (char**)calloc(MaxSize, sizeof(char*));
		*returnSize = 0;
		int sta[MaxSize];	//stack
		ConstructPaths(root, paths, returnSize, sta, 0);
		return paths;
	}
	char** BinaryTreePaths_BFS(struct TreeNode* root, int* returnSize) {
		char** paths = (char**)calloc(MaxSize, sizeof(char*));
		*returnSize = 0;
		if (!root) return paths;
		struct TreeNode** node_queue = (struct TreeNode**)calloc(MaxSize, sizeof(struct TreeNode*));
		char** path_queue = (char**)calloc(MaxSize, sizeof(char*));
		int left = 0, right = 0;
		char* tmp = (char*)calloc(MaxSize, sizeof(char));
		sprintf(tmp, "%d", root->val);
		node_queue[right] = root;
		path_queue[right++] = tmp;
		while (left < right)
		{
			struct TreeNode* node = node_queue[left];
			char* path = path_queue[left++];
			if (!node->left && !node->right)
				paths[(*returnSize)++] = path;
			else
			{
				int n = strlen(path);
				if (node->left)
				{
					tmp = (char*)calloc(MaxSize, sizeof(char));
					for (int i = 0; i < n; ++i)
						tmp[i] = path[i];
					sprintf(tmp + n, "->%d", node->left->val);
					node_queue[right] = node->left;
					path_queue[right++] = tmp;
				}
				if (node->right)
				{
					tmp = (char*)calloc(MaxSize, sizeof(char));
					for (int i = 0; i < n; ++i)
						tmp[i] = path[i];
					sprintf(tmp + n, "->%d", node->right->val);
					node_queue[right] = node->right;
					path_queue[right++] = tmp;
				}
			}
		}
		return paths;
	}
	void ConstructPaths_Backtracking(struct TreeNode* root, char* path, char** paths, int* returnSize)
	{
		int len = strlen(path); //中
		sprintf(path + len, "%d", root->val);
		if (!root->left && !root->right)
		{
			paths[(*returnSize)++] = path;
			return;
		}
		if (root->left)
		{
			char* tmp = (char*)calloc(MaxSize, sizeof(char));
			strcpy(tmp, path);
			strcat(tmp, "->");	//直接strcat()会修改原串导致错误,这里"->"体现回溯
			ConstructPaths_Backtracking(root->left, tmp, paths, returnSize);//左
			//每次函数调用完，path依然是没有加上"->" 的，这就是回溯了
		}
		if (root->right)
		{
			char* tmp = (char*)calloc(MaxSize, sizeof(char));
			strcpy(tmp, path);
			strcat(tmp, "->");
			ConstructPaths_Backtracking(root->right, tmp, paths, returnSize);//右
			//因为并有没有改变path的数值，执行完递归函数之后，path依然是之前的数值（相当于回溯了）
		}
	}
	char** BinaryTreePaths_Backtracking(struct TreeNode* root, int* returnSize) {
		char** paths = (char**)calloc(MaxSize, sizeof(char*));
		*returnSize = 0;
		if (!root) return paths;
		char* path = (char*)calloc(MaxSize, sizeof(char));
		ConstructPaths_Backtracking(root, path, paths, returnSize);
		return paths;
	}
	char** BinaryTreePaths_Stack(struct TreeNode* root, int* returnSize) 
	{
		char** paths = (char**)calloc(MaxSize, sizeof(char*));
		*returnSize = 0;
		if (!root) return paths;
		struct TreeNode* treeSt[MaxSize];   // 保存树的遍历节点
		char* pathSt[MaxSize];  // 保存遍历路径的节点
		int top_t = 0, top_p = 0;
		pathSt[top_p] = (char*)calloc(MaxSize, sizeof(char));
		treeSt[top_t++] = root;
		sprintf(pathSt[top_p++], "%d", root->val);
		while (top_t)
		{
			struct TreeNode* node = treeSt[--top_t];  // 取出节点 中
			char* path = pathSt[--top_p];   // 取出该节点对应的路径
			if (!node->left && !node->right)    // 遇到叶子节点
				paths[(*returnSize)++] = path;
			if (node->right)
			{
				treeSt[top_t++] = node->right;
				char* tmp = (char*)calloc(MaxSize, sizeof(char));
				sprintf(tmp, "%s->%d", path, node->right->val);
				pathSt[top_p++] = tmp;
			}
			if (node->left)
			{
				treeSt[top_t++] = node->left;
				char* tmp = (char*)calloc(MaxSize, sizeof(char));
				sprintf(tmp, "%s->%d", path, node->left->val);
				pathSt[top_p++] = tmp;
			}
		}
		return paths;
	}
	int SumOfLeftLeaves(struct TreeNode* root) {
		if (!root) return 0;
		int midVal = 0;
		//如果左节点不为空，且左节点没有左右孩子，那么这个节点就是左叶子
		//判断当前节点是不是左叶子是无法判断的，必须要通过节点的父节点来判断其左孩子是不是左叶子
		if (root->left && !root->left->left && !root->left->right)  //该结点左孩子为左叶子
			midVal = root->left->val;
		// 递归检查左右子树
		return midVal + SumOfLeftLeaves(root->left) + SumOfLeftLeaves(root->right);
	}
	int SumOfLeftLeaves_Iteration(struct TreeNode* root) {
		if (!root) return 0;
		int sum = 0;
		struct TreeNode* Stack[MaxSize];
		int top = 0;
		Stack[top++] = root;
		while (top)
		{
			struct TreeNode* node = Stack[--top];
			if (node->left && !node->left->left && !node->left->right)
				sum += node->left->val;
			if (node->right) Stack[top++] = node->right;
			if (node->left) Stack[top++] = node->left;
		}
		return sum;
	}	
	void Search(TreeNode* T, TreeNode* (&q), ElemType key)//查找值域等于data的结点，存在则q指向该结点
	{
		if (!T) return;
		if (T->val == key) q = T;
		else
		{
			Search(T->left, q, key);
			if (q == NULL)	//剪枝操作，在左子树未找到才到右子树中查找
				Search(T->right, q, key);
		}
	}
	bool IsSymmetricHelper(TreeNode* left, TreeNode* right)
	{
		//如果左右节点都为空，说明当前节点是叶子节点，返回true
		if (left == NULL && right == NULL) return true;
		//如果当前结点只有一个子节点或者有两个值不同的子节点，返回false
		if (left == NULL || right == NULL || left->val != right->val)
			return false;
		//然后左子节点的左子节点和右子节点的右子节点比较，左子节点的右子节点和右子节点的左子节点比较
		return IsSymmetricHelper(left->left, right->right) && IsSymmetricHelper(left->right, right->left);
	}
	bool IsSymmetric(TreeNode* T)
	{
		if (!T) return true;
		return IsSymmetricHelper(T->left, T->right);
	}
	bool IsSymmetric_Queue(TreeNode* root)
	{
		if (!root) return true;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = root->left;   //将左子树结点入队
		rear = (rear + 1) % MaxSize;
		queue[rear] = root->right;   //将右子树结点入队
		while (front != rear)
		{
			front = (front + 1) % MaxSize;
			TreeNode* leftNode = queue[front];
			front = (front + 1) % MaxSize;
			TreeNode* rightNode = queue[front];
			if (!leftNode && !rightNode)   //左，右结点均为空，此时说明是对称的
				continue;
			//左右一个结点不为空，或者都不为空但值不同，返回false
			if (!leftNode || !rightNode || leftNode->val != rightNode->val)
				return false;
			rear = (rear + 1) % MaxSize;
			queue[rear] = leftNode->left;   //将左结点左孩子入队
			rear = (rear + 1) % MaxSize;
			queue[rear] = rightNode->right;   //将右结点右孩子入队
			rear = (rear + 1) % MaxSize;
			queue[rear] = leftNode->right;   //将左结点右孩子入队
			rear = (rear + 1) % MaxSize;
			queue[rear] = rightNode->left;   //将右结点左孩子入队
		}
		return true;
	}
	bool IsSymmetric_Stack(TreeNode* root)
	{
		if (!root) return true;
		TreeNode* Stack[MaxSize];
		int top = 0;
		Stack[top++] = root->left;
		Stack[top++] = root->right;
		while (top)
		{
			TreeNode* leftNode = Stack[--top];
			TreeNode* rightNode = Stack[--top];
			if (!leftNode && !rightNode)   //左，右结点均为空，此时说明是对称的
				continue;
			//左右一个结点不为空，或者都不为空但值不同，返回false
			if (!leftNode || !rightNode || leftNode->val != rightNode->val)
				return false;
			Stack[top++] = leftNode->left;  //将左结点左孩子入队
			Stack[top++] = rightNode->right;  //将右结点右孩子入队
			Stack[top++] = leftNode->right;  //将左结点右孩子入队
			Stack[top++] = rightNode->left;  //将右结点左孩子入队
		}
		return true;
	}
	bool IsSameTree(TreeNode* p, TreeNode* q)
	{
		if (!p && !q) return true;
		else if (!p || !q) return false;
		else if (p->val != q->val) return false;
		else return IsSameTree(p->left, q->left) && IsSameTree(p->right, q->right);
	}
	bool IsSameTree_Queue(TreeNode* p, TreeNode* q)
	{
		if (!p && !q) return true;
		else if (!p || !q) return false;
		TreeNode* queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear] = p;
		rear = (rear + 1) % MaxSize;
		queue[rear] = q;
		while (front != rear)
		{
			front = (front + 1) % MaxSize;
			TreeNode* leftNode = queue[front];
			front = (front + 1) % MaxSize;
			TreeNode* rightNode = queue[front];
			if (!leftNode && !rightNode)
				continue;
			if (!leftNode || !rightNode || leftNode->val != rightNode->val)
				return false;
			rear = (rear + 1) % MaxSize;
			queue[rear] = leftNode->left;
			rear = (rear + 1) % MaxSize;
			queue[rear] = rightNode->left;
			rear = (rear + 1) % MaxSize;
			queue[rear] = leftNode->right;
			rear = (rear + 1) % MaxSize;
			queue[rear] = rightNode->right;
		}
		return true;
	}
	bool IsSubtree(TreeNode* root, TreeNode* subRoot) {
		if (!subRoot) return true;
		if (!root) return false;
		return IsSameTree(root, subRoot) || IsSubtree(root->left, subRoot) || IsSubtree(root->right, subRoot);
	}
	//KMP解法,即将树前序遍历序列进行字符串匹配(带NULL)
	int GetMaxElement(TreeNode* root)
	{
		if (!root) return INT_MIN;
		int lmax = GetMaxElement(root->left);
		int rmax = GetMaxElement(root->right);
		int tmp = max(lmax, rmax);
		return max(root->val, tmp);
		//return max(root->val, max(GetMaxElement(root->left), GetMaxElement(root->right)));
		//警告：千万别这么写，会超时，因为宏展开求解了2遍
	}
	void GetDFSOrder(TreeNode* root, int* array, int* returnSize, int lnull, int rnull)
	{
		if (!root)
		{
			return;
		}
		array[(*returnSize)++] = root->val;
		if (root->left) {
			GetDFSOrder(root->left, array, returnSize, lnull, rnull);
		}
		else array[(*returnSize)++] = lnull;
		if (root->right) {
			GetDFSOrder(root->right, array, returnSize, lnull, rnull);
		}
		else array[(*returnSize)++] = rnull;
	}
	void GetNext(int* next, int* substr, int len)
	{
		int t1 = 0, t2 = -1;
		next[0] = -1;
		while (t1 < len)
		{
			if (t2 == -1 || substr[t1] == substr[t2])
				next[++t1] = ++t2;
			else t2 = next[t2];
		}
	}
	int KMP(int* str, int* substr, int len, int sublen)
	{
		if (!sublen) return 0;
		int t1 = 0, t2 = 0;
		int* next = (int*)calloc(sublen + 1, sizeof(int));	//注意+1
		GetNext(next, substr, sublen);
		while (t1 < len)
		{
			if (t2 == -1 || str[t1] == substr[t2])
				t1++, t2++;
			else t2 = next[t2];
			if (t2 == sublen) return t1 - sublen;	//此为数组下标,若为实际index,则+1
		}
		return -1;
	}
	bool IsSubtree_KMP(TreeNode* root, TreeNode* subRoot)
	{
		int rootMaxval = GetMaxElement(root);
		int subRootMaxval = GetMaxElement(subRoot);
		int maxElement = max(rootMaxval, subRootMaxval);
		//int maxElement = max(GetMaxElement(root), GetMaxElement(subRoot));
		//警告：千万别这么写，会超时，因为宏展开求解了2遍
		int lnull = maxElement + 1, rnull = maxElement + 2;
		int* rootOrder = (int*)calloc(MaxSize, sizeof(int));
		int* subRootOrder = (int*)calloc(MaxSize, sizeof(int));
		int len = 0, sublen = 0;
		GetDFSOrder(root, rootOrder, &len, lnull, rnull);
		GetDFSOrder(subRoot, subRootOrder, &sublen, lnull, rnull);
		if (KMP(rootOrder, subRootOrder, len, sublen) + 1) return true;
		return false;
	}
	
	bool HasPathSum(TreeNode* T, int targetSum)   
	{
		//BFS,判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和targetSum
		//用栈DFS同理,栈中同时保存了 (节点，路径和)
		//也就是说只要能把所有的节点访问一遍，那么就一定能找到正确的结果
		if (!root) return false;
		typedef struct PathNode
		{
			struct TreeNode* node;
			int pathSum;
		}PathNode;
		struct PathNode queue[MaxSize];
		int front = 0, rear = 0;
		rear = (rear + 1) % MaxSize;
		queue[rear].node = root;
		queue[rear].pathSum = root->val;
		int cnt = 0;
		while (front != rear)
		{
			cnt = (rear + MaxSize - front) % MaxSize;
			for (int i = 0; i < cnt; i++)
			{
				front = (front + 1) % MaxSize;
				PathNode p = queue[front];
				if (!p.node->left && !p.node->right)
					if (p.pathSum == targetSum) return true;
				if (p.node->left)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear].node = p.node->left;
					queue[rear].pathSum = p.pathSum + p.node->left->val;
				}
				if (p.node->right)
				{
					rear = (rear + 1) % MaxSize;
					queue[rear].node = p.node->right;
					queue[rear].pathSum = p.pathSum + p.node->right->val;
				}
			}
		}
		return false;
	}
	bool HasPathSum_Recursion(struct TreeNode* root, int targetSum) {
		if (!root) return false;
		if (!root->left && !root->right && targetSum == root->val)
			return true;
		//可以用递减，然后每次减去遍历路径节点上的数值
		return HasPathSum_Recursion(root->left, targetSum - root->val) || HasPathSum_Recursion(root->right, targetSum - root->val);
	//回溯隐藏在这里,因为把targetSum - root->val 直接作为参数传进去，函数结束，count的数值没有改变
	}
	void TraversePathSum(struct TreeNode* cur, int targetSum, int* returnSize, int** returnColumnSizes, int** ans, int* path, int pathSize)
	{
		//大回溯 + 指针传值
		if (!cur->left && !cur->right)
		{
			if (!targetSum)
			{
				(*returnColumnSizes)[*returnSize] = pathSize;
				ans[(*returnSize)++] = path;
			}
			return;
		}
		if (cur->left)
		{
			int* tmp = (int*)calloc(pathSize + 1, sizeof(int));
			memcpy(tmp, path, pathSize * sizeof(int));
			tmp[pathSize] = cur->left->val;
			TraversePathSum(cur->left, targetSum - cur->left->val, returnSize, returnColumnSizes, ans, tmp, pathSize + 1);
		}
		if (cur->right)
		{
			int* tmp = (int*)calloc(pathSize + 1, sizeof(int));
			memcpy(tmp, path, pathSize * sizeof(int));
			tmp[pathSize] = cur->right->val;
			TraversePathSum(cur->right, targetSum - cur->right->val, returnSize, returnColumnSizes, ans, tmp, pathSize + 1);
		}
	}
	int** PathSumAll(struct TreeNode* root, int targetSum, int* returnSize, int** returnColumnSizes) 
	{
		int** ans = (int**)calloc(MaxSize, sizeof(int*));
		*returnSize = 0;
		if (!root) return ans;
		int* path = (int*)calloc(MaxSize, sizeof(int));
		path[0] = root->val;	// 把根节点放进路径
		(*returnColumnSizes) = (int*)calloc(MaxSize, sizeof(int));
		TraversePathSum(root, targetSum - root->val, returnSize, returnColumnSizes, ans, path, 1);
		return ans;
	}

	ElemType GetLeft(TreeNode* T)
	{
		TreeNode* temp = T;
		while (temp->left) temp = temp->left;
		return temp->val;
	}
	ElemType GetRight(TreeNode* T)
	{
		TreeNode* temp = T;
		while (temp->right) temp = temp->right;
		return temp->val;
	}
	bool JudgeBiSortTree(TreeNode* T)	//二叉排序树的判定
	{
		if (T == NULL) return true;
		else
		{
			if (T->left && T->val <= GetRight(T->left)) return false;
			if (T->right && T->val > GetLeft(T->right)) return false;
			return JudgeBiSortTree(T->left) && JudgeBiSortTree(T->right);
		}
	}
};


//8.ThreadBiTree(线索二叉树)
typedef struct TBTNode   //线索二叉树定义
{
	char data;
	int ltag, rtag;   //线索标记
	struct TBTNode* lchild;
	struct TBTNode* rchild;
}TBTNode;
class ThreadBiTree
{
private:
	void InOrderThread(TBTNode* T, TBTNode* (&pre))	//中序遍历对二叉树线索化(配合中序遍历使用)
	{
		if (!T) return;
		InOrderThread(T->lchild, pre);   //递归，左子树线索化
		if (!T->lchild)   //建立当前结点的前驱线索
		{
			T->lchild = pre;
			T->ltag = 1;
		}
		if (pre && pre->rchild == NULL)   //建立前驱结点的后继线索
		{
			pre->rchild = T;
			pre->rtag = 1;
		}
		pre = T;   //pre指向当前的T，作为T将要指向的下一个结点的前驱结点指示指针
		InOrderThread(T->rchild, pre);   //指向T的右子树，此时pre与T的右子树分别指向的结点形成了一个前驱后继对
										 //为下一次线索的连接做准备，递归线索化右子树
	}
public:
	TBTNode* root;
	ThreadBiTree():root(NULL)
	{}
	~ThreadBiTree()
	{}
	void CreateInOrderThreadTree(TBTNode* T)	//通过中序遍历建立中序线索二叉树(已整合的函数)
	{
		TBTNode* pre = NULL;   //前驱结点指针
		if (!T) return;
		InOrderThread(T, pre);   //非空二叉树，线索化
		pre->rchild = NULL;
		pre->rtag = 1;   //后处理中序最后一个结点
	}
};


//9.AGraph(图的邻接表实现)
#define maxSize 100
typedef struct ArcNode
{
	int adjvex;   //该边所指向的结点的位置
	struct ArcNode* nextarc;   //指向下一条边的指针
	int weight;   //(权值)，无要求时可缺省
}ArcNode;//边结点
typedef struct
{
	char data;   //顶点信息
	ArcNode* firstarc;   //指向第一条边的指针
}VNode;//顶点结点
class AGraph
{
private:
	VNode adjlist[maxSize];   //邻接表
	int n, e;   //顶点数和边数
	int visit[maxSize];
	//visit[]数组，作为顶点的访问标记，初始时所有的元素均为0，表示所有顶点都未被访问
	//因图中可能存在回路，当前经过的顶点在将来还可能再次经过，所以要对每个顶点进行标记，以免重复访问	
public:
	AGraph() :n(0), e(0), adjlist(), visit()
	{}
	AGraph(int n, int e) :n(n), e(e), adjlist(), visit()
	{}
	~AGraph()
	{}
	void Visit(int num)
	{
		printf("%d ", num);
	}
	void AddArc(int head, int tail, int weight)
	{
		if (!adjlist[head].firstarc)
		{
			adjlist[head].firstarc = (ArcNode*)malloc(sizeof(ArcNode));
			adjlist[head].firstarc->adjvex = tail;
			adjlist[head].firstarc->weight = weight;
			adjlist[head].firstarc->nextarc = NULL;
			return;
		}
		ArcNode* p = adjlist[head].firstarc, * q = p->nextarc;
		if (p->adjvex > tail)
		{
			ArcNode* t = (ArcNode*)malloc(sizeof(ArcNode));
			t->adjvex = tail;
			t->weight = weight;
			t->nextarc = p;
			adjlist[head].firstarc = t;
		}
		while (q && q->adjvex < tail)
		{
			p = p->nextarc;
			q = p->nextarc;
		}
		if (!q)
		{
			q = (ArcNode*)malloc(sizeof(ArcNode));
			q->adjvex = tail;
			q->weight = weight;
			q->nextarc = NULL;
			p->nextarc = q;
		}
		else
		{
			ArcNode* t = (ArcNode*)malloc(sizeof(ArcNode));
			t->adjvex = tail;
			t->weight = weight;
			p->nextarc = t;
			t->nextarc = q;
		}
	}
	void SetParameter(int n, int e)
	{
		this->n = n;
		this->e = e;
	}
	void Create()
	{
		for (int i = 1; i <= e; i++)
		{
			int head, tail, weight;
			scanf("%d%d%d", &head, &tail, &weight);
			AddArc(head, tail, weight);
		}
	}
	void ResetTraverse()
	{
		memset(visit, 0, maxSize);
	}
	void DFStraverse(int v)	//v是起点编号,深搜遍历(连通图)
	{
		ArcNode* p;
		visit[v] = 1;   //置已访问标记
		Visit(v);   //函数Visit()代表了一类访问结点v的操作
		p = adjlist[v].firstarc;   //p指向顶点v的第一条边
		while (p)
		{
			if (visit[p->adjvex] == 0)   //若顶点未访问，则递归访问它
				DFStraverse(p->adjvex);
			p = p->nextarc;   //p指向顶点v的下一条边的终点
		}
	}
	void BFStraverse(int v)	//广搜遍历(连通图)
	{
		ArcNode* p;
		int que[maxSize], front = 0, rear = 0;   //这是队列定义的简单写法
		int j;
		Visit(v);   //任意访问顶点v的函数
		visit[v] = 1;
		rear = (rear + 1) % maxSize;   //当前顶点v入队
		que[rear] = v;
		while (front != rear)   //队空的时候说明遍历完成
		{
			front = (front + 1) % maxSize;   //顶点出队
			j = que[front];
			p = adjlist[j].firstarc;   //p指向出队顶点j的第一条边
			while (p)   //将p的所有邻接点中未被访问的入队
			{
				if (visit[p->adjvex] == 0)   //当前邻接顶点未被访问，则入队
				{
					Visit(p->adjvex);
					visit[p->adjvex] = 1;
					rear = (rear + 1) % maxSize;   //该顶点入队
					que[rear] = p->adjvex;
				}
				p = p->nextarc;   //p指向j的下一条边
			}
		}
	}
	void DFSdisTraverse()	//非连通图
	{
		for (int i = 0; i < n; i++)
			if (visit[i] == 0)
				DFStraverse(i);
	}
	void BFSdisTraverse()
	{
		for (int i = 0; i < n; i++)
			if (visit[i] == 0)
				BFStraverse(i);
	}
	void Show()
	{
		for (int i = 1; i <= n; i++)
		{
			printf("%d ", i);
			ArcNode* p = adjlist[i].firstarc;
			while (p)
			{
				printf("%d ", p->adjvex);
				p = p->nextarc;
			}
			putchar('\n');
		}
	}
	int RemotePoint(int v)	//求不带权无向连通图G中距离v最远的一个顶点，即路径长度最长
	{
		ArcNode* p;
		int que[maxSize], front = 0, rear = 0;
		int visit[maxSize];
		int j = 0;
		memset(visit, 0, sizeof(visit));
		rear = (rear + 1) % maxSize;
		que[rear] = v;
		visit[v] = 1;
		while (front != rear)
		{
			front = (front + 1) % maxSize;
			j = que[front];
			p = adjlist[j].firstarc;
			while (p)
			{
				if (visit[p->adjvex] == 0)
				{
					visit[p->adjvex] = 1;
					rear = (rear + 1) % maxSize;
					que[rear] = p->adjvex;
				}
				p = p->nextarc;
			}
		}
		return j;   //队空时，j保存了遍历过程中的最后一个顶点
	}
};//图的邻接表类型


//10.MGraph(图的邻接矩阵实现)
#define maxSize 100
typedef struct
{
	int no;   //顶点编号
	char info;   //顶点其他信息
}VertexType;   //顶点类型
class MGraph  //图的定义
{
private:
	int edges[maxSize][maxSize];   // 邻接矩阵定义
	int n, e;   //n顶点数，e边数
	VertexType vex[maxSize];   //存放结点信息
	int IsVisited[maxSize];//标记顶点是否访问
	void Visit(int val)
	{
		printf("%d ", val);
	}
	void DFSTraverseHelper(int val)
	{
		if (IsVisited[val]) return;
		Visit(val);
		IsVisited[val] = 1;
		for (int i = 0; i < n; i++)
			if (edges[val][i]) DFSTraverseHelper(i);
	}
	void BFSTraverseHelper(int val)
	{
		int que[maxSize], front = 0, rear = 0;   //队列的简单写法
		int j;
		Visit(val);
		IsVisited[val] = 1;
		rear = (rear + 1) % maxSize;   //当前顶点入队
		que[rear] = val;
		while (front != rear)   //队空时说明遍历完成
		{
			front = (front + 1) % maxSize;   //顶点出队
			j = que[front];
			for (int i = j + 1; i < n; i++)   //将所有邻接点中未被访问的入队
			{
				if (edges[j][i])
					if (!IsVisited[i])
					{
						Visit(i);
						IsVisited[i] = 1;
						rear = (rear + 1) % maxSize;
						que[rear] = i;
					}
			}
		}
	}
public:
	MGraph() :edges(), n(0), e(0), vex(), IsVisited()
	{}
	~MGraph()
	{}
	void InitMGraph()
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				edges[i][j] = 0;
			vex[i].no = i;
		}
	}
	void InputMGraph()
	{
		int m, n;
		for (int i = 0; i < e; i++)
		{
			scanf("%d%d", &m, &n);
			edges[m][n] = 1;
			edges[n][m] = 1;
		}
	}
	void ShowMGraph()
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				printf("%d ", edges[i][j]);
			putchar('\n');
		}
	}
	void DFSTraverse()
	{
		for (int i = 0; i < n; i++)
			if (!IsVisited[i])
				DFSTraverseHelper(i);
	}
	void BFSTraverse()
	{
		for (int i = 0; i < n; i++)
			if (!IsVisited[i])
				BFSTraverseHelper(i);
	}
};	//图的邻接矩阵类型


//11.String(字符串)
class String
{
private:
	char* s;
	int length;
public:
	String() :s(NULL), length(0)
	{}
	String(int n) :length(n)
	{
		s = (char*)malloc(n * sizeof(char));
	}
	~String()
	{}
	void Assign(const char* str)
	{
		if (s)
			free(s);	//释放原串空间
		int len = 0;
		const char* c = str;
		while (*c)
		{
			len++;
			c++;
		}
		if (len == 0)
		{
			s = NULL;
			length = 0;
		}
		else
		{
			s = (char*)malloc(sizeof(char) * (len + 1));
			c = str;
			for (int i = 0; i <= len; i++, c++)
			{
				s[i] = *c;
			}
			length = len;
		}
	}
	void Print(void)
	{
		printf("%s", s);
	}
	void Clear(void)
	{
		if (!s) return;
		free(s);
		s = NULL;
		length = 0;
	}
	int Strlen(void)
	{
		return length;
	}
	int Strcmp(String Sp)
	{
		int pos = 0;
		if (!length && !Sp.length) return 0;
		if (!length) return -Sp.s[pos];
		if (!Sp.length) return s[pos];
		int cmp = s[pos] - Sp.s[pos];
		while (!cmp)
		{
			pos++;
			if (pos > length || pos > Sp.length)
				return 0;
		}
		return cmp;
	}
	void Strcat(String Sp)
	{
		if (!Sp.s) return;
		int prelen = length;
		length += Sp.length;
		s = (char*)realloc(s, (length + 1) * sizeof(char));
		for (int i = 0; i <= Sp.length; i++)
			s[i + prelen] = Sp.s[i];
	}
	String SubString(int pos, int len)
	{
		String temp;
		if (pos < 0 || pos >= length || len<0 || pos + len>length)
			return temp;
		temp.s = (char*)malloc(sizeof(char) * (len + 1));
		for (int i = 0; i < len; i++)
			temp.s[i] = s[pos + i];
		temp.s[len] = 0;
		temp.length = len;
		return temp;
	}
	void GetNext(int next[])
	{
		next[0] = next[1] = 0;
		int k = 0;
		for (int i = 1; i < length; i++)
		{
			while (k && s[i] != s[k])
				k = next[k];
			next[i + 1] = s[i] == s[k] ? ++k : 0;
		}
	}
	int KMP(String substr, int next[])
	{
		int k = 0;
		for (int i = 0; i < length; i++)
		{
			while (k && s[i] != substr.s[k])
				k = next[k];
			if (s[i] == substr.s[k]) k++;
			if (k == substr.length) return i - substr.length + 2;
		}
		return -1;
	}
	void GetNext_While(int* next)
	{
		int t1 = 0, t2 = -1;
		next[0] = -1;
		while (t1 < length)
		{
			if (t2 == -1 || s[t1] == s[t2])
				next[++t1] = ++t2;
			else t2 = next[t2];
		}
	}
	int KMP_While(String substr, int* next)
	{
		int t1 = 0, t2 = 0;
		while (t1 < length)
		{
			if (t2 == -1 || s[t1] == substr.s[t2])
				t1++, t2++;
			else t2 = next[t2];
			if (t2 == substr.length) return t1 - substr.length + 1;
			//printf("%d\n", t1 - substr.length + 1), t2 = next[t2];//持续匹配
		}
		return -1;
	}
};


//12.Queue(队列)
typedef int ElemType;
#define LENG sizeof(Queue)  //结点所占单元数
class Queue
{
private:
	ElemType data;
	Queue* next;
public:
	Queue() :data(0), next(NULL)
	{}
	~Queue()
	{}
	void Create(void)
	{
		ElemType val;
		Queue* temp, * tail = this;
		cin >> val;
		while (val)
		{
			temp = (Queue*)malloc(LENG);
			temp->data = val;
			temp->next = NULL;
			tail->next = temp;
			tail = tail->next;
			data++;
			cin >> val;
		}
	}
	void Print(void)
	{
		Queue* p = next;
		while (p)
		{
			cout << p->data << " ";
			p = p->next;
		}
		putchar('\n');
	}
	void EnQueue(ElemType t)
	{
		Queue* tail = this;
		while (tail->next)
			tail = tail->next;
		Queue* temp = (Queue*)malloc(LENG);
		temp->data = t;
		temp->next = NULL;
		tail->next = temp;
		data++;
	}
	ElemType DeQueue(void)
	{
		Queue* del = next;
		if (!del)
		{
			printf("队列为空，无法出队\n");
			return 0;
		}
		next = del->next;
		ElemType t = del->data;
		free(del);
		del = NULL;
		data--;
		return t;
	}
	bool Empty(void)
	{
		return !next;
	}
	void Clear()
	{
		while (data)
			DeQueue();
		next = NULL;
	}
	class StackByQueue
	{
	private:
		Queue* Q;
	public:
		void InitStack(void)
		{
			Q = (Queue*)calloc(1, sizeof(Queue));
			Q->data = 0;
			Q->next = NULL;
		}
		void Push(ElemType val)
		{
			Q->EnQueue(val);
		}
		ElemType Pop(void)
		{
			int cnt = Q->data;
			while (--cnt)
			{
				Q->EnQueue(Q->DeQueue());
			}
			return Q->DeQueue();
		}
		ElemType GetTop(void)
		{
			Queue* temp = Q;
			while (temp->next)
			{
				temp = temp->next;
			}
			return temp->data;
		}
		bool Empty(void)
		{
			return !Q->data;
		}
		void Clear(void)
		{
			while (Q->data) Q->DeQueue();
			free(Q);
		}


	}StackInQueue;


};


//13.Deque & MonoQueue(双端队列 & 单调队列)
typedef int ElemType;
#define LENG sizeof(Deque)  //结点所占单元数
class Deque
{
private:
	ElemType data;
	Deque* head, * tail;
public:
	Deque() :data(0), head(NULL), tail(NULL)
	{}
	~Deque()
	{}
	void Push_front(ElemType val)
	{
		Deque* temp = (Deque*)calloc(1, LENG);
		temp->data = val;
		if (!data)
		{
			head = temp;
			tail = temp;
			temp->tail = NULL;
			temp->head = NULL;
		}
		else
		{
			temp->tail = head;
			head->head = temp;
			head = temp;
			temp->head = NULL;
		}
		data++;
	}
	ElemType Pop_front(void)
	{
		if (!head) return 0;
		Deque* del = head;
		head = del->tail;
		ElemType val = del->data;
		free(del);
		data--;
		return val;
	}
	void Push_back(ElemType val)
	{
		Deque* temp = (Deque*)calloc(1, LENG);
		temp->data = val;
		if (!data)
		{
			head = temp;
			tail = temp;
			temp->tail = NULL;
			temp->head = NULL;
		}
		else
		{
			temp->head = tail;
			tail->tail = temp;
			tail = temp;
			temp->tail = NULL;
		}
		data++;
	}
	ElemType Pop_back(void)
	{
		if (!tail) return 0;
		Deque* del = tail;
		tail = del->head;
		ElemType val = del->data;
		free(del);
		data--;
		return val;
	}
	bool Empty(void)
	{
		return !data;
	}
	void Clear(void)
	{
		Deque* del = head;
		while (del)
		{
			Deque* next = del->tail;
			free(del);
			del = next;
		}
		data = 0;
		head = NULL;
		tail = NULL;
	}
	ElemType Front(void)
	{
		if (head) return head->data;
		printf("ERROR!\n");
		return -1;
	}
	ElemType Back(void)
	{
		if (tail) return tail->data;
		printf("ERROR!\n");
		return -1;
	}
};
class MonoQueueDown//单调队列（从大到小）
{
private:
	Deque que;// 使用deque来实现单调队列
public:
	// 每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
	// 同时pop之前判断队列当前是否为空。
	void Pop(int value)
	{
		if (!que.Empty() && value == que.Front())
			que.Pop_front();
	}
	// 如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，
	// 直到push的数值小于等于队列入口元素的数值为止。
	// 这样就保持了队列里的数值是单调从大到小的了。
	void Push(int value)
	{
		while (!que.Empty() && value > que.Back())
			que.Pop_back();
		que.Push_back(value);
	}
	// 查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
	int Front()
	{
		return que.Front();
	}
	ElemType* MaxSlidingWindow(ElemType* nums, int numsSize, int k, int* returnSize)
	{
		*returnSize = numsSize - k + 1;
		ElemType* ans = (ElemType*)calloc(*returnSize, sizeof(int));
		//ans[]存储k大小的滑动窗口中最大值或最小值
		for (int i = 0; i < k; i++)//先将前k的元素放进队列
			Push(nums[i]);
		int pos = 0;
		ans[pos++] = Front(); // ans 记录前k的元素的最大值
		for (int i = k; i < numsSize; i++)
		{
			Pop(nums[i - k]); // 滑动窗口移除最前面元素
			Push(nums[i]); // 滑动窗口前加入最后面的元素
			ans[pos++] = Front();
		}
		return ans;
	}
};
class MonoQueueUp//单调队列（从小到大）
{
private:
	Deque que;// 使用deque来实现单调队列
public:
	// 每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
	// 同时pop之前判断队列当前是否为空。
	void Pop(int value)
	{
		if (!que.Empty() && value == que.Front())
			que.Pop_front();
	}
	// 如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，
	// 直到push的数值小于等于队列入口元素的数值为止。
	// 这样就保持了队列里的数值是单调从小到大的了。
	void Push(int value)
	{
		while (!que.Empty() && value < que.Back())
			que.Pop_back();
		que.Push_back(value);
	}
	// 查询当前队列里的最小值 直接返回队列前端也就是front就可以了。
	int Front()
	{
		return que.Front();
	}
	ElemType* MinSlidingWindow(ElemType* nums, int numsSize, int k, int* returnSize)
	{
		*returnSize = numsSize - k + 1;
		ElemType* ans = (ElemType*)calloc(*returnSize, sizeof(int));
		//ans[]存储k大小的滑动窗口中最大值或最小值
		for (int i = 0; i < k; i++)//先将前k的元素放进队列
			Push(nums[i]);
		int pos = 0;
		ans[pos++] = Front(); // ans 记录前k的元素的最大值
		for (int i = k; i < numsSize; i++)
		{
			Pop(nums[i - k]); // 滑动窗口移除最前面元素
			Push(nums[i]); // 滑动窗口前加入最后面的元素
			ans[pos++] = Front();
		}
		return ans;
	}
};


//14.HPInt (High-Precision Integer)
class HPInt
{
private:
	bool minus;
	vector<int> num;
public:
	HPInt() :minus(false)
	{
		num.emplace_back(0);
	}
	HPInt(int n) :minus(false)
	{
		*this = n;
	}
	HPInt(const string& s) :minus(false)
	{
		*this = s;
	}
	HPInt(HPInt& b) :minus(b.minus)
	{
		int size = b.num.size();
		for (int i = 0; i < size; ++i)
			num.emplace_back(b.num[i]);
	}
	HPInt(const HPInt& b) :minus(b.minus)
	{
		int size = b.num.size();
		for (int i = 0; i < size; ++i)
			num.emplace_back(b.num[i]);
	}
	HPInt(HPInt&& b) noexcept :minus(b.minus)
	{
		num = (vector<int>&&)b.num;
	}
	~HPInt()
	{}
	HPInt& operator = (const string& s)
	{
		num.clear();
		int cnt = s.size();
		if (s[0] == '-')
		{
			minus = true;
			for (int i = 0; i < cnt - 1; ++i)
				num.emplace_back(s[cnt - 1 - i] - '0');
		}
		else
		{
			minus = false;
			for (int i = 0; i < cnt; ++i)
				num.emplace_back(s[cnt - 1 - i] - '0');
		}
		return *this;
	}
	HPInt& operator = (int n)
	{
		string s = to_string(n);
		*this = s;
		return *this;
	}
	HPInt& operator = (const HPInt& b)
	{
		num.clear();
		minus = b.minus;
		int size = b.num.size();
		for (int i = 0; i < size; ++i)
		{
			num.emplace_back(b.num[i]);
		}
		return *this;
	}
	string GetNum() const
	{
		string res = "";
		if (minus) res.push_back('-');
		for (int i = num.size() - 1; i >= 0; --i)
			res.push_back(num[i] + '0');
		if (res == "") res = "0";
		return res;
	}
	HPInt abs() const
	{
		HPInt c(*this);
		c.minus = false;
		return c;
	}
	HPInt absoluteValueSum(const HPInt& b) const
	{
		HPInt c;
		c.minus = false;
		c.num.clear();
		int acnt = num.size(), bcnt = b.num.size();
		int maxcnt = max(acnt, bcnt);
		for (int i = 0, g = 0; g || i < maxcnt; ++i)
		{
			int x = g;  //g is for cheking carry
			if (i < acnt) x += num[i];
			if (i < bcnt) x += b.num[i];
			c.num.emplace_back(x % 10);
			g = x / 10;
		}
		return c;
	}
	HPInt absoluteValueDiffer(const HPInt& b) const
	{
		HPInt absa = abs(), absb = b.abs();
		if (absa < absb) return b.absoluteValueDiffer(*this);
		HPInt c;
		c.minus = false;
		c.num.clear();
		int acnt = absa.num.size(), bcnt = absb.num.size();
		int maxcnt = max(acnt, bcnt);
		for (int i = 0, g = 0; g || i < maxcnt; ++i)
		{
			int x = g;  //g is for cheking carry
			if (i < acnt) x += absa.num[i];
			if (i < bcnt) x -= absb.num[i];
			if (x < 0)
			{
				x += 10;
				g = -1;
			}
			else g = 0;
			c.num.emplace_back(x);
		}
		int pos = num.size() - 1;
		while (c.num[pos] == 0 && pos)
		{
			c.num.pop_back();
			--pos;
		}
		return c;
	}
	HPInt pow(const HPInt& k) const
	{
		if (k == 0) return 1;
		else if (k % 2 == 1)
			return pow(k - 1) * *this;
		else
		{
			HPInt temp = pow(k / 2);
			return temp * temp;
		}
	}

	HPInt operator + (const HPInt& b) const
	{
		if (!minus && b.minus)
		{
			HPInt c = absoluteValueDiffer(b);
			if (*this < b.abs())
			{
				c.minus = true;
			}
			return c;
		}
		else if (minus && !b.minus)
		{
			HPInt c = absoluteValueDiffer(b);
			if (abs() > b)
			{
				c.minus = true;
			}
			return c;
		}
		HPInt c = absoluteValueSum(b);
		c.minus = minus;
		return c;
	}
	HPInt& operator += (const HPInt& b)
	{
		*this = *this + b;
		return *this;
	}
	HPInt operator - (const HPInt& b) const
	{
		HPInt oppb(b);
		oppb.minus = !oppb.minus;
		return *this + oppb;
	}
	HPInt& operator -= (const HPInt& b)
	{
		*this = *this - b;
		return *this;
	}
	HPInt operator * (const HPInt& b) const
	{
		HPInt c;
		if ((minus + b.minus) & 1) c.minus = true;
		c.num.clear();
		int alen = num.size(), blen = b.num.size();
		c.num.resize(alen + blen);
		for (int i = 0; i < alen; ++i)
			for (int j = 0; j < blen; ++j)
				c.num[i + j] += num[i] * b.num[j];  //simulate Column Multiplication
		int len = alen + blen;
		for (int i = 0; i < len; ++i)
			if (c.num[i] > 9)   //carry
			{
				c.num[i + 1] += c.num[i] / 10;
				c.num[i] %= 10;
			}
		while (c.num[len - 1] == 0 && len > 1)
		{
			c.num.pop_back();
			--len;
		}
		return c;
	}
	HPInt& operator *= (const HPInt& b)
	{
		*this = *this * b;
		return *this;
	}
	HPInt operator / (const HPInt& b) const
	{
		HPInt c;
		c.num.clear();
		if (abs() < b)
		{
			c = "0";
			return c;
		}
		HPInt tempa(abs());
		HPInt absb = b.abs();
		string initstr = absb.GetNum();
		string quotient = "";
		while (tempa >= initstr + "0")
			initstr += "0";
		while (absb <= initstr)
		{
			int curQuotient = 0;
			while (tempa >= initstr)
			{
				tempa -= initstr;
				++curQuotient;
			}
			quotient.push_back(curQuotient + '0');
			initstr.pop_back();
		}
		c = quotient;
		if ((minus + b.minus) & 1) c.minus = true;
		return c;
	}
	HPInt& operator /= (const HPInt& b)
	{
		*this = *this / b;
		return *this;
	}
	HPInt operator % (const HPInt& b) const
	{
		HPInt c = *this / b;
		c = *this - b * c;
		return c;
	}
	HPInt& operator %= (const HPInt& b)
	{
		*this = *this % b;
		return *this;
	}

	bool operator < (const HPInt& b) const
	{
		if (minus)
		{
			if (!b.minus) return true;
			if (num.size() != b.num.size()) return num.size() > b.num.size();
			for (int i = num.size() - 1; i >= 0; --i)
				if (num[i] != b.num[i]) return num[i] > b.num[i];
			return false;
		}
		else
		{
			if (b.minus) return false;
			if (num.size() != b.num.size()) return num.size() < b.num.size();
			for (int i = num.size() - 1; i >= 0; --i)
				if (num[i] != b.num[i]) return num[i] < b.num[i];
			return false;
		}
	}
	bool operator > (const HPInt& b) const
	{
		return b < *this;
	}
	bool operator <= (const HPInt& b) const
	{
		return !(b < *this);
	}
	bool operator >= (const HPInt& b) const
	{
		return !(*this < b);
	}
	bool operator != (const HPInt& b) const
	{
		return b < *this || *this < b;
	}
	bool operator == (const HPInt& b) const
	{
		return !(b < *this) && !(*this < b);
	}

};
istream& operator >> (istream& in, HPInt& x)
{
	string s;
	in >> s;
	x = s;
	return in;
}
ostream& operator << (ostream& out, HPInt& x)
{
	out << x.GetNum();
	return out;
}




/*                         DataStructureByArray(基于数组实现各种数据结构)                             */


//1.DifferenceArray(差分数组)
class DifferenceArray
{
private:
	int len;
	int* Dnums;
public:
	DifferenceArray(int n) : len(n), Dnums((int*)calloc(n + 1, sizeof(int)))
	{}
	~DifferenceArray()
	{}
	void Initialize(int* nums)
	{
		for (int i = 1; i <= len; i++)
			Dnums[i] = nums[i] - nums[i - 1];
	}
	void IntervalAdd(int l, int r, int append)
	{
		//下标1起始
		Dnums[l] += append;	//0起始:nums[l - 1]
		Dnums[r + 1] -= append;	//0起始:nums[r]
	}
	int* OriginalArray(int* returnSize)
	{
		*returnSize = len;
		int* ans = (int*)calloc(len + 1, sizeof(int));
		//下标1起始
		for (int i = 1; i <= len; i++)
			ans[i] = ans[i - 1] + Dnums[i];
		//下标0起始: ans[0] = Dnums[0]; ans[i] = ans[i - 1] + Dnums[i]
		return ans;
	}
};


//2.Stack(栈)
class Stack
{
private:
	int top;
	int* stack;
public:
	Stack(int n) :top(0), stack((int*)calloc(n, sizeof(int)))
	{}
	~Stack()
	{}
	void Push(int val)
	{
		stack[top++] = val;
	}
	int Pop(void)
	{
		return stack[--top];
	}
	bool Empty(void)
	{
		return top ? true : false;
	}
	void Clear(void)
	{
		while (top)
			Pop();
	}
	int Peak(void)
	{
		return stack[top - 1];
	}
	int* MonoStack(int* nums, int len, const char* op)
	{
		for (int i = 1; i <= len; ++i)
			if (!strcmp(op, "up"))
			{
				while (top && nums[stack[top - 1]] < nums[i])
					Pop();
			}

	}
};





/*									  库函数,关键字,宏  												*/


//1.calloc()	分配所需的内存空间，并返回一个指向它的指针,会设置分配的内存为零
#include <stdlib.h>
//void* calloc(size_t count, size_t size);
void calloc_test(void)
{
	int* a = (int*)calloc(5, sizeof(int));
}


//2.memset()	数组批量处理函数
#include <string.h>
void memset_test(void)
{
	char str[50];
	//全处理
	memset(str, 0, sizeof(char) * 50);
	memset(str, '#', sizeof(str));
	//部分处理
	memset(str, '$', 7);	
	
	int num[20];
	memset(num, 0x3f, sizeof(num));//作为无穷大使用(0x3f3f3f3f)且可以保证无穷大加无穷大仍然不会超限(INT_MAX)
	//0x7fffffff = INT_MAX, 0x3f3f3f3f(10^9级别)   即0x7f,0x3f

	double fnum[10];
	memset(fnum, 127, sizeof(fnum));//近似无限大
	memset(fnum, 0, sizeof(fnum));//清零

	//不能用memset处理动态数组申请和指针传参
}


//3.sizeof	特殊的编译预处理
void sizeof_test(int* c)
{
	//可见数组名可获得整个数组大小
	int a[10];
	sizeof(a) == sizeof(int) * 10 == 40;

	//动态申请及传参指针均视为指针大小==4
	int* b = (int*)malloc(sizeof(int) * 10);
	sizeof(b) == sizeof(int*) == 4;

	sizeof(c) == sizeof(int*) == 4;//不管c是数组名或是动态申请的数组
}


//4.INT_MAX & INT_MIN
#include <limits.h>
#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)


//5.memcmp()	类似strcmp(),按字节比较函数
#include <string.h>
void memcmp_test(void)
{
	char a[4] = "abc";
	char b[4] = "ace";
	memcmp(a, b, 1);	// = 'a' - 'a' = 0;
	memcmp(a, b, 3);	// = 'b' - 'c' = -1, 不会比较第三个
	//对于memcmp()，如果两个字符串相同而且count大于字符串长度的话，
	//memcmp不会在\0处停下来，会继续比较\0后面的内存单元，直到_res不为零或者达到count次数
	//想使用memcmp比较字符串，要保证count不能超过最短字符串的长度，否则结果有可能是错误的
	int c[3] = { 1 , 2 , 3 };
	int d[3] = { 1 , 2 , 5 };
	//注意是按字节比较:sizeof(int) = 4, 按int的每个字节比较,很不稳定
	memcmp(c, d, 4);	// = 0
	memcmp(c, d, 8);	// = 0
	memcmp(c, d, 9);	// < 0 (-1)
	memcmp(c, d, 12);	// < 0 (-1)
	int num1 = 5, num2 = 3;
	memcmp(&num1, &num2, 4);	// > 0 (1)
	//int型似乎就 1 0 -1,分别表示 > = <

}


//6.qsort(C语言快排)
#include <stdlib.h>
int cmp(const void* a, const void* b)
{
	return *(int*)a - *(int*)b;
	//返回值 < 0, a在b左边
	//返回值 = 0, 位置不确定
	//返回值 > 0, a在b右边
	//结构体 （*(struct*)a).member -（*(struct*)b).member
}
void qsort_test(void)
{
	int num[10];
	//qsort(nums, numsSize, sizeof(int), cmp);
	qsort(num, 10, sizeof(int), cmp);	
}


//7.memcpy(复制数组类)
#include <string.h>
void memcpy_test(void)
{
	int a[4] = { 1, 2, 2, 3 };
	int b[4];
	memcpy(b, a, 4 * sizeof(int));
	//类比strcpy(),最后一个参数为字节数
}





/*										   some tips												*/


//1.OperationOnLinearTable(线性表操作小程序)
/*#define _CRT_SECURE_NO_WARNINGS
#define gets gets_s

#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>
#include<string.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
typedef int status;
typedef int ElemType; //数据元素类型定义
#define LIST_INIT_SIZE 100
#define LISTINCREMENT  10
typedef int ElemType;

typedef struct {  //顺序表（顺序结构）的定义
	ElemType* elem;
	int length;
	int listsize;
}SqList;

typedef struct {  //线性表的集合类型定义
	struct {
		char name[30];
		SqList L;
	} elem[10];
	int length;
}LISTS;
LISTS Lists;      //线性表集合的定义Lists


status InitList(SqList& L);   //构造一个空的线性表L
status DestroyList(SqList& L);   //销毁线性表L
status ClearList(SqList& L);   //清空线性表
status ListEmpty(SqList L);   //判断线性表L是否为空
status ListLength(SqList L);   //求线性表L的长度
status GetElem(SqList L, int i, ElemType& e);   //获取线性表L的第i个元素
status LocateElem(SqList L, ElemType e);   //查找元素e在线性表L中的位置序号
status PriorElem(SqList L, ElemType e, ElemType& pre);   //获取线性表L中元素e的前驱
status NextElem(SqList L, ElemType e, ElemType& next);   //获取线性表L中元素e的后继
status ListInsert(SqList& L, int i, ElemType e);   //将元素e插入到线性表L的第i个元素之前
status ListDelete(SqList& L, int i, ElemType& e);   //删除线性表L的第i个元素
status ListTraverse(SqList L);   //依次显示线性表中的元素
status ListInput(SqList& L, int n);   //输入元素组成线性表
status ListInput(SqList& L);   //输入元素组成线性表
status AddList(LISTS& Lists, char ListName[]);//在Lists中增加一个名称为ListName的线性表
status RemoveList(LISTS& Lists, char ListName[]);  //Lists中删除一个名称为ListName的线性表
int LocateList(LISTS Lists, char ListName[]); // 在Lists中查找一个名称为ListName的线性表，成功返回逻辑序号，否则返回0
status ShowList(LISTS Lists);  //显示多线性表中的各元素
status ClearLists(LISTS& Lists);   //清空多线性表
status ListsEmpty(LISTS Lists);   //判断多线性表是否为空表
int ListsLength(LISTS Lists);   //得到多线性表长度
status GetList(int i, LISTS Lists);   //得到多线性表的第i号线性表
status GetList(char s[], LISTS Lists);   //得到多线性表中名称为所给名称的线性表
status PriorList(LISTS Lists, char ListName[]);   //查找线性表前驱
status NextList(LISTS Lists, char ListName[]);   //查找线性表后继
status ListInsert(LISTS& Lists, int i, SqList& L, char ListName[]);   //在多线性表指定位置插入线性表
status MultipleListsInput(LISTS& Lists);   //一次性输入多个线性表进入多线性表
status SaveList(SqList L, char FileName[]);   //将线性表L的全部元素写入到文件名为FileName的文件中
status LoadList(SqList& L, char FileName[]);   //将文件名为FileName的数据读入到线性表L中

status InitList(SqList& L)   //线性表L不存在，构造一个空的线性表，返回OK，否则返回INFEASIBLE。
{
	if (L.elem)  return INFEASIBLE;  //线性表L若已存在，返回INFEASIBLE
	L.elem = (ElemType*)malloc(LIST_INIT_SIZE * sizeof(ElemType));   //存储空间分配
	if (!L.elem) exit(OVERFLOW);   //存储分配失败
	L.length = 0;
	L.listsize = LIST_INIT_SIZE;
	return OK;
}

status DestroyList(SqList& L) //如果线性表L存在，销毁线性表L，释放数据元素的空间，返回OK，否则返回INFEASIBLE。
{
	if (L.elem)
	{
		free(L.elem);   //释放空间
		L.elem = NULL;  //指针置空
		L.length = 0;
		L.listsize = 0;   //线型表长度，大小重置
		return OK;
	}
	else return INFEASIBLE;
}

status ClearList(SqList& L)//如果线性表L存在，删除线性表L中的所有元素，返回OK，否则返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表未构造
	free(L.elem);
	L.elem = (ElemType*)malloc(LIST_INIT_SIZE * sizeof(ElemType));   //重新分配空间
	L.length = 0;
	L.listsize = LIST_INIT_SIZE;
	return OK;
}

status ListEmpty(SqList L)//如果线性表L存在，判断线性表L是否为空，空就返回TRUE，否则返回FALSE；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	if (L.length) return FALSE;   //不为空表
	return TRUE;   //为空表
}

status ListLength(SqList L)//如果线性表L存在，返回线性表L的长度，否则返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	return L.length;   //返回线性表L的长度
}

status GetElem(SqList L, int i, ElemType& e)// 如果线性表L存在，获取线性表L的第i个元素，保存在e中，返回OK；如果i不合法，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	if (i<1 || i>L.length) return ERROR;   //位置i不合法
	e = L.elem[i - 1];   //保存值于e中
	return OK;
}

status LocateElem(SqList L, ElemType e)//初始条件是线性表已存在；操作结果是返回L中第1个与e相等的数据元素的位序，若这样的数据元素不存在，则返回值为0。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	for (int i = 0; i < L.length; i++)   //遍历查找
		if (e == L.elem[i]) return i + 1;
	return ERROR;   //e不存在
}

status PriorElem(SqList L, ElemType e, ElemType& pre)//如果线性表L存在，获取线性表L中元素e的前驱，保存在pre中，返回OK；如果没有前驱，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	for (int i = 1; i < L.length; i++)   //遍历线性表，查找元素e
		if (e == L.elem[i]) { pre = L.elem[i - 1]; return OK; }   //若找到，获取线性表L中元素e的前驱
	return ERROR;   //e没有前驱
}

status NextElem(SqList L, ElemType e, ElemType& next)//如果线性表L存在，获取线性表L元素e的后继，保存在next中，返回OK；如果没有后继，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	for (int i = 0; i < L.length - 1; i++)   //遍历线性表，查找元素e
		if (e == L.elem[i]) { next = L.elem[i + 1]; return OK; }   //若找到，获取线性表L中元素e的后继
	return ERROR;   //e没有后继
}

status ListInsert(SqList& L, int i, ElemType e)//如果线性表L存在，将元素e插入到线性表L的第i个元素之前，返回OK；当插入位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	if (i<1 || i>L.length + 1) return ERROR;   //插入位置不正确
	if (L.length >= L.listsize)   //存储空间不够
	{
		ElemType* newbase;   //创建新地址
		newbase = (ElemType*)realloc(L.elem, (L.listsize + LISTINCREMENT) * sizeof(ElemType));   //扩充存储空间
		if (newbase == NULL) return ERROR;   //扩充失败
		L.elem = newbase;   //更新基地址
		L.listsize += LISTINCREMENT;   //更新线性表容量
	}
	for (int j = L.length - 1; j >= i - 1; j--)   //向后移动元素，空出第i个元素的位置elem[i-1]
		L.elem[j + 1] = L.elem[j];
	L.elem[i - 1] = e;   //插入元素
	L.length++;   //更新长度
	return OK;
}

status ListDelete(SqList& L, int i, ElemType& e)//如果线性表L存在，删除线性表L的第i个元素，并保存在e中，返回OK；当删除位置不正确时，返回ERROR；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	if (i<1 || i>L.length) return ERROR;   //删除位置不正确
	e = L.elem[i - 1];   //保存第i个元素于e中
	for (int k = i - 1; k < L.length - 1; k++)   //删除该元素
		L.elem[k] = L.elem[k + 1];
	L.length--;   //更新长度
	return OK;
}

status ListTraverse(SqList L)//如果线性表L存在，依次显示线性表中的元素，每个元素间空一格，返回OK；如果线性表L不存在，返回INFEASIBLE。
{
	if (!L.elem) return INFEASIBLE;   //线性表L不存在
	if (!L.length) return ERROR;   //长度为0，返回错误
	printf("\n-----------all elements -----------------------\n");
	for (int i = 0; i < L.length - 1; i++)
	{
		printf("%d ", L.elem[i]);   //依次显示线性表
		if (i + 1 == L.length - 1)   //最后一个元素特判，不需要后置空格
		{
			printf("%d", L.elem[i + 1]);
		}
	}
	printf("\n------------------ end ------------------------\n");
	return OK;
}

status ListInput(SqList& L, int n)
{
	if (!L.elem) return INFEASIBLE;
	while (n >= L.listsize)   //溢出时扩充
	{
		ElemType* newbase;   //创建新地址
		newbase = (ElemType*)realloc(L.elem, (L.listsize + LISTINCREMENT) * sizeof(ElemType));
		if (newbase == NULL) return ERROR;   //扩充失败
		L.elem = newbase;   //更新基地址
		L.listsize += LISTINCREMENT;
	}
	printf("请依次输入元素，空格隔开：\n");
	for (int i = 0; i < n; i++)
		scanf("%d", &L.elem[i]);
	L.length = n;
	return OK;
}

status ListInput(SqList& L)
{
	ElemType elem;
	int i = 0;
	printf("请输入数据，以0结束！\n");
	while (scanf("%d", &elem) == 1 && elem)
	{
		if (L.length >= L.listsize)
		{
			ElemType* newbase = (ElemType*)realloc(L.elem, (L.listsize + LISTINCREMENT) * sizeof(ElemType));
			if (!newbase) exit(OVERFLOW);
			L.elem = newbase;
			L.listsize += LISTINCREMENT;
		}
		L.elem[i++] = elem;
		L.length++;
	}
	return OK;
}

status AddList(LISTS& Lists, char ListName[]) //在Lists中增加一个名称为ListName的线性表
{
	if (Lists.length == 10) return ERROR;
	int i;
	InitList(Lists.elem[Lists.length].L);
	for (i = 0; ListName[i] != 0; i++)
		Lists.elem[Lists.length].name[i] = ListName[i];
	Lists.elem[Lists.length].name[i] = 0;
	ListInput(Lists.elem[Lists.length].L);
	Lists.length++;
	return OK;
}

status RemoveList(LISTS& Lists, char ListName[])
{
	int flag = 0;   //判断是否进行了删除
	int* p = (int*)malloc(Lists.length * sizeof(int));   //p数组用于判断是否需要删除
	for (int i = 0; i < Lists.length; i++) p[i] = 1;
	for (int k = 0; k < Lists.length; k++)
	{
		for (int i = 0; ListName[i] != 0; i++)
		{
			if (Lists.elem[k].name[i] != ListName[i]) { p[k] = 0; break; }  //名称不同，不需要删除
		}
	}
	for (int i = 0; i < Lists.length; i++)
		if (p[i])   //需要删除
		{
			for (int k = i; k < Lists.length - 1; k++)
			{
				for (int j = 0; Lists.elem[k + 1].name[j] != 0; j++)  //下一线性表名称替换需要删除的线性表名称
					Lists.elem[k].name[j] = Lists.elem[k + 1].name[j];
				DestroyList(Lists.elem[i].L);
				InitList(Lists.elem[i].L);
				for (int j = 0; j < Lists.elem[k + 1].L.length; j++)  //元素替换
					ListInsert(Lists.elem[k].L, j + 1, Lists.elem[k + 1].L.elem[j]);
				Lists.elem[k].L.length = Lists.elem[k + 1].L.length;
			}
			flag = 1;
			Lists.length--;
		}
	if (flag) return OK;
	return ERROR;
}

int LocateList(LISTS Lists, char ListName[]) // 在Lists中查找一个名称为ListName的线性表，成功返回逻辑序号，否则返回0
{
	int* p = (int*)malloc(Lists.length * sizeof(int));  //p数组用于记录是否同名
	for (int i = 0; i < Lists.length; i++) p[i] = 1;
	for (int k = 0; k < Lists.length; k++)
	{
		for (int i = 0; ListName[i] != 0; i++)
		{
			if (Lists.elem[k].name[i] != ListName[i]) { p[k] = 0; break; }
		}
	}
	for (int i = 0; i < Lists.length; i++)
		if (p[i]) return i + 1;
	return 0;
}

status ShowList(LISTS Lists)
{
	if (!Lists.length) return ERROR;   //长度为0，返回错误
	printf("\n-----------all elements -----------------------\n");
	for (int i = 0; i < Lists.length; i++)
	{
		printf("%s   ", Lists.elem[i].name);
		for (int j = 0; j < Lists.elem[i].L.length; j++)
			printf("%d ", Lists.elem[i].L.elem[j]);
		putchar('\n');
	}
	printf("\n------------------ end ------------------------\n");
	return OK;
}

status ClearLists(LISTS& Lists)
{
	Lists.length = 0;
	return OK;
}

status ListsEmpty(LISTS Lists)
{
	if (Lists.length) return TRUE;
	else return FALSE;
}

int ListsLength(LISTS Lists)
{
	return Lists.length;
}

status GetList(int i, LISTS Lists)
{
	if (i<1 || i>Lists.length) return ERROR;
	printf("多线性表第%d个线性表为：\n", i);
	printf("%s   ", Lists.elem[i - 1].name);
	for (int k = 0; k < Lists.elem[i - 1].L.length; k++)
		printf("%d ", Lists.elem[i - 1].L.elem[k]);
	putchar('\n');
	return OK;
}

status GetList(char s[], LISTS Lists)
{
	int i = LocateList(Lists, s);
	if (!i) return ERROR;
	printf("多线性表中名称为\"%s\"的线性表为：\n", s);
	printf("%s   ", Lists.elem[i - 1].name);
	for (int k = 0; k < Lists.elem[i - 1].L.length; k++)
		printf("%d ", Lists.elem[i - 1].L.elem[k]);
	putchar('\n');
	return OK;
}

status PriorList(LISTS Lists, char ListName[])
{
	for (int i = 0; i < Lists.length - 1; i++)
	{
		if (strcmp(Lists.elem[i + 1].name, ListName) == 0)
		{
			printf("线性表\"%s\"的前驱是：\n", ListName);
			printf("%s ", Lists.elem[i].name);
			for (int j = 0; j < Lists.elem[i].L.length; j++)
			{
				printf("%d", Lists.elem[i].L.elem[j]);
				if (j < Lists.elem[i].L.length - 1) printf(" ");
			}
			return OK;
		}
	}
	printf("查找失败！\n");
	return ERROR;
}

status NextList(LISTS Lists, char ListName[])
{
	for (int i = 0; i < Lists.length - 1; i++)
	{
		if (strcmp(Lists.elem[i].name, ListName) == 0)
		{
			printf("线性表\"%s\"的后继是：\n", ListName);
			printf("%s ", Lists.elem[i + 1].name);
			for (int j = 0; j < Lists.elem[i + 1].L.length; j++)
			{
				printf("%d", Lists.elem[i + 1].L.elem[j]);
				if (j < Lists.elem[i + 1].L.length - 1) printf(" ");
			}
			return OK;
		}
	}
	printf("查找失败！\n");
	return ERROR;
}

status ListInsert(LISTS& Lists, int i, SqList& L, char ListName[])
{
	if (Lists.length >= 10)
	{
		return OVERFLOW;
	}
	if (i<1 || i>L.length + 1) return ERROR;
	InitList(Lists.elem[Lists.length].L);
	for (int j = Lists.length; j >= i; j--)
	{
		strcpy(Lists.elem[j].name, Lists.elem[j - 1].name);
		for (int k = 0; k < Lists.elem[j - 1].L.length; k++)
			Lists.elem[j].L.elem[k] = Lists.elem[j - 1].L.elem[k];
		Lists.elem[j].L.length = Lists.elem[j - 1].L.length;
		Lists.elem[j].L.listsize = Lists.elem[j - 1].L.listsize;
	}
	strcpy(Lists.elem[i - 1].name, ListName);
	for (int k = 0; k < L.length; k++)
		Lists.elem[i - 1].L.elem[k] = L.elem[k];
	Lists.elem[i - 1].L.length = L.length;
	Lists.length++;
	return OK;
}

status MultipleListsInput(LISTS& Lists)
{
	int state;
	int i = 0;
	printf("输入多个线性表及其内部数据：\n");
	printf("输入1开始！\n");
	while (scanf("%d", &state) == 1 && state == 1)
	{
		if (Lists.length >= 10)
		{
			printf("多线性表已满，不能再继续输入数据！");
			return OVERFLOW;
		}
		printf("请输入线性表名称：\n");
		scanf("%s", Lists.elem[i].name);
		ListInput(Lists.elem[i].L);
		i++;
		Lists.length++;
		printf("是否继续？1：继续输入；-1：结束输入！\n");
	}
	return OK;
}

status SaveList(SqList L, char FileName[])
{
	FILE* fp;
	if ((fp = fopen(FileName, "w")) == NULL) return ERROR;
	//fwrite(L.elem, sizeof(ElemType), L.length, fp);
	for (int i = 0; i < L.length; i++)
		fprintf(fp, "%d ", L.elem[i]);
	fclose(fp);
	return OK;
}

status LoadList(SqList& L, char FileName[])
{
	if (L.length) return INFEASIBLE;
	L.length = 0;
	FILE* fp;
	if ((fp = fopen(FileName, "r")) == NULL) return ERROR;
	//while (fread(&L.elem[L.length], sizeof(ElemType), 1, fp))
	//	L.length++;
	while (fscanf(fp, "%d ", &L.elem[L.length++]) != EOF);
	L.length--;
	fclose(fp);
	return OK;
}
*/


//2.OperationOnLNodeTable(链表操作小程序)
/*#define _CRT_SECURE_NO_WARNINGS
#define gets gets_s

// Linear Table On Sequence Structure 
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
//------------------------------------
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASTABLE -1
#define INFEASIBLE -1
#define OVERFLOW -2

typedef int status;
typedef int ElemType; //数据元素类型定义
#define LIST_INIT_SIZE 100
#define LISTINCREMENT  10
typedef struct LNode {  //单链表（链式结构）结点的定义
	ElemType data;
	struct LNode* next;
}LNode, * LinkList;
typedef struct {  //线性链表的集合类型定义
	struct {
		char name[30];
		LinkList L;
	} elem[10];
	int length;
}LISTS;
LISTS Lists;      //线性链表集合的定义Lists


status InitList(LinkList& L);
status DestroyList(LinkList& L);
status ClearList(LinkList& L);
status ListEmpty(LinkList L);
status ListLength(LinkList L);
status GetElem(LinkList L, int i, ElemType& e);
status LocateElem(LinkList L, ElemType e);
status PriorElem(LinkList L, ElemType cur, ElemType& pre_e);
status NextElem(LinkList L, ElemType cur, ElemType& next_e);
status ListInsert(LinkList& L, int i, ElemType e);
status ListDelete(LinkList& L, int i, ElemType& e);
status ListTraverse(LinkList L);
status ListInput(LinkList& L);
status MulLists_Input(LISTS& Lists);
status AddList(LISTS& Lists, char ListName[]);
status RemoveList(LISTS& Lists, char ListName[]);
status Clear_or_Ini_Lists(LISTS& Lists);
status ListsEmpty(LISTS Lists);
status ListsLength(LISTS Lists);
status GetList(LISTS Lists, int i);
status PriorList(LISTS Lists, char ListName[]);
status NextList(LISTS Lists, char ListName[]);
status ListInsert1(LISTS& Lists, int i, LinkList L1, char ListName[]);
status ListsTraverse(LISTS Lists);
void Operation_For_Single_List(void);
int LocateList(LISTS Lists, char ListName[]);
void File_Reading_And_Writing(void);
status SaveList(LinkList L, char FileName[]);
status LoadList(LinkList& L, char FileName[]);



int main(void) {
	int op1 = 1;
	ElemType i;
	char Lname[30];
	LinkList L1 = NULL;
	while (op1) {
		system("cls");
		printf("\n\n");
		printf("                    Menu for MultiLinear Table Management \n");
		printf("---------------------------------------------------------------------------------------\n");
		printf("    	  1. Clear_or_Ini_Lists     8. RemoveList\n");
		printf("    	  2. ListsEmpty             9. LocateList\n");
		printf("    	  3. ListsLength            10. ListsTraverse\n");
		printf("    	  4. GetList                11. AddList\n");
		printf("    	  5. PriorList              12. MulLists_Input\n");
		printf("    	  6. NextList               13. Operation_For_Single_List\n");
		printf("          7. ListInsert             14. Operation_For_File_Reading_And_Writing\n");
		printf("    	  0. Exit\n");
		printf("---------------------------------------------------------------------------------------\n");
		printf("        请选择你的操作[0~14]:\n");
		scanf("%d", &op1);
		switch (op1) {
		case 1:
			if (Clear_or_Ini_Lists(Lists) == OK) printf("多线性链表清空或初始化成功！\n");
			else printf("多线性链表清空或初始化失败！\n");
			getchar(); getchar();
			break;
		case 2:
			if (ListsEmpty(Lists) == TRUE) printf("多线性链表为空！");
			else printf("多线性链表非空！");
			getchar(); getchar();
			break;
		case 3:
			printf("多线性链表个数为%d", ListsLength(Lists));
			getchar(); getchar();
			break;
		case 4:
			printf("请输入要获取的线性链表的序号：");
			scanf("%d", &i);
			GetList(Lists, i);
			getchar(); getchar();
			break;
		case 5:
			printf("请输入要查找前驱的线性链表的名称：");
			scanf("%s", Lname);
			PriorList(Lists, Lname);
			getchar(); getchar();
			break;
		case 6:
			printf("请输入要查找后继的线性链表的名称：");
			scanf("%s", Lname);
			NextList(Lists, Lname);
			getchar(); getchar();
			break;
		case 7:
			printf("请输入要插入的线性链表的名称：\n");
			scanf("%s", Lname);
			InitList(L1);
			ListInput(L1);
			printf("请输入要插入的位置：\n");
			scanf("%d", &i);
			if (ListInsert1(Lists, i, L1, Lname) == OK) printf("线性链表%s插入成功！\n", Lname);
			getchar(); getchar();
			break;
		case 8:
			printf("请输入要删除的线性链表的名称：\n");
			scanf("%s", Lname);
			if (RemoveList(Lists, Lname) == OK) printf("线性链表%s删除成功！\n", Lname);
			else printf("多线性链表中无%s，线性链表删除失败！\n", Lname);
			getchar(); getchar();
			break;
		case 9:
			printf("请输入要定位的线性链表的名称：\n");
			scanf("%s", Lname);
			if (!LocateList(Lists, Lname)) printf("未查找到%s！\n", Lname);
			else printf("%s的逻辑序号为：%d\n", Lname, LocateList(Lists, Lname));
			getchar(); getchar();
			break;
		case 10:
			if (ListsTraverse(Lists) != OK) printf("多线性链表为空！");
			getchar(); getchar();
			break;
		case 11:
			printf("请输入要增加的空线性链表的名称：\n");
			scanf("%s", Lname);
			if (AddList(Lists, Lname) == OK) printf("空线性链表%s增加成功！\n", Lname);
			else printf("空线性链表%s增加失败！\n", Lname);
			getchar(); getchar();
			break;
		case 12:
			if (MulLists_Input(Lists) == OK) printf("数据输入成功！\n");
			else printf("数据输入失败！\n");
			getchar(); getchar();
			break;
		case 13:
			Operation_For_Single_List();
			getchar(); getchar();
			break;
		case 14:
			File_Reading_And_Writing();
			getchar(); getchar();
			break;
		case 0:
			break;
		default:
			printf("无此操作，请重新选择操作数！\n");
			break;
		}//end of switch
	}//end of while
	printf("欢迎下次再使用本系统！\n");
}//end of main()



status InitList(LinkList& L)
// 线性链表L不存在，构造一个空的线性链表，返回OK，否则返回INFEASIBLE。
{
	if (!L)
	{
		L = (LNode*)malloc(sizeof(LNode));
		L->next = NULL;
		return OK;
	}
	return INFEASIBLE;
}

status DestroyList(LinkList& L)
// 如果线性链表L存在，销毁线性链表L，释放数据元素的空间，返回OK，否则返回INFEASIBLE。
{
	LNode* p;
	if (L)
	{
		while (L)
		{
			p = L;
			L = L->next;
			free(p);
		}
		return OK;
	}
	return INFEASIBLE;
}

status ClearList(LinkList& L)
// 如果线性链表L存在，删除线性链表L中的所有元素，返回OK，否则返回INFEASIBLE。
{
	LNode* p, * q;
	if (L)
	{
		p = L->next;
		while (p)
		{
			q = p;
			p = p->next;
			free(q);
		}
		L->next = NULL;
		return OK;
	}
	return INFEASIBLE;
}

status ListEmpty(LinkList L)
// 如果线性链表L存在，判断线性链表L是否为空，空就返回TRUE，否则返回FALSE；如果线性链表L不存在，返回INFEASIBLE。
{
	if (L)
	{
		if (L->next) return FALSE;
		return TRUE;
	}
	return INFEASIBLE;
}


int ListLength(LinkList L)
// 如果线性链表L存在，返回线性链表L的长度，否则返回INFEASIBLE。
{
	if (L)
	{
		int length = 0;
		LNode* p = L->next;
		while (p)
		{
			length++;
			p = p->next;
		}
		return length;
	}
	return INFEASIBLE;
}


status GetElem(LinkList L, int i, ElemType& e)
// 如果线性链表L存在，获取线性链表L的第i个元素，保存在e中，返回OK；如果i不合法，返回ERROR；如果线性链表L不存在，返回INFEASIBLE。
{
	if (L)
	{
		if (i < 1) return ERROR;
		LNode* p = L;
		while (p && i)
		{
			p = p->next;
			i--;
		}
		if (i) return ERROR;
		e = p->data;
		return OK;
	}
	return INFEASIBLE;
}


status LocateElem(LinkList L, ElemType e)
// 如果线性链表L存在，查找元素e在线性链表L中的位置序号；如果e不存在，返回ERROR；当线性链表L不存在时，返回INFEASIBLE。
{
	if (L)
	{
		int col = 0;
		LNode* p = L->next;
		while (p)
		{
			col++;
			if (e == p->data)
				return col;
			p = p->next;
		}
		return ERROR;
	}
	return INFEASIBLE;
}


status PriorElem(LinkList L, ElemType e, ElemType& pre)
// 如果线性链表L存在，获取线性链表L中元素e的前驱，保存在pre中，返回OK；如果没有前驱，返回ERROR；如果线性链表L不存在，返回INFEASIBLE。
{
	if (L)
	{
		LNode* p, * q;
		p = L->next; q = L;
		while (p)
		{
			if (p->data == e && q != L)
			{
				pre = q->data;
				return OK;
			}
			p = p->next;
			q = q->next;
		}
		return ERROR;
	}
	return INFEASIBLE;
}

status NextElem(LinkList L, ElemType e, ElemType& next)
// 如果线性链表L存在，获取线性链表L元素e的后继，保存在next中，返回OK；如果没有后继，返回ERROR；如果线性链表L不存在，返回INFEASIBLE。
{
	if (L)
	{
		LNode* p, * q;
		q = L->next;
		if (q == NULL) return ERROR;
		p = L->next->next;
		while (p)
		{
			if (q->data == e)
			{
				next = p->data;
				return OK;
			}
			p = p->next;
			q = q->next;
		}
		return ERROR;
	}
	return INFEASIBLE;
}


status ListInsert(LinkList& L, int i, ElemType e)
// 如果线性链表L存在，将元素e插入到线性链表L的第i个元素之前，返回OK；当插入位置不正确时，返回ERROR；如果线性链表L不存在，返回INFEASIBLE。
{
	LNode* p;
	int j = 0;
	if (L)
	{
		p = L;
		while (p && j < i - 1)
		{
			p = p->next;
			j++;
		}
		if (!p || i < 1) return ERROR;
		LNode* q = (LNode*)malloc(sizeof(LNode));
		q->data = e;
		q->next = p->next;
		p->next = q;
		return OK;
	}
	return INFEASIBLE;
}


status ListDelete(LinkList& L, int i, ElemType& e)
// 如果线性链表L存在，删除线性链表L的第i个元素，并保存在e中，返回OK；当删除位置不正确时，返回ERROR；如果线性链表L不存在，返回INFEASIBLE。
{
	LNode* p = L, * q;
	int j = 0;
	if (L)
	{
		while (p && j < i - 1)
		{
			p = p->next;
			j++;
		}
		if (!p || i < 1) return ERROR;
		q = p->next;
		e = q->data;
		p->next = p->next->next;
		free(q);
		return OK;
	}
	return INFEASIBLE;
}


status ListTraverse(LinkList L)
// 如果线性链表L存在，依次显示线性链表中的元素，每个元素间空一格，返回OK；如果线性链表L不存在，返回INFEASIBLE。
{
	if (L)
	{
		LNode* p = L->next;
		while (p)
		{
			printf("%d", p->data);
			if (p->next) printf(" ");
			p = p->next;
		}
		return OK;
	}
	return INFEASIBLE;
}

status ListInput(LinkList& L)
//输入数据至L中
{
	ElemType elem;
	printf("请输入数据，以-1结束！\n");
	LinkList p = L;
	while (scanf("%d", &elem) == 1 && elem != -1)
	{
		p->next = (LinkList)malloc(sizeof(LNode));
		p = p->next;
		p->data = elem;
		p->next = NULL;
	}
	return OK;
}

status AddList(LISTS& Lists, char ListName[])
//在Lists中增加一个名称为ListName的空线性链表
{
	InitList(Lists.elem[Lists.length].L);
	int i;
	for (i = 0; ListName[i] != '\0'; i++)
		Lists.elem[Lists.length].name[i] = ListName[i];
	Lists.elem[Lists.length].name[i] = '\0';
	Lists.length++;
	return OK;
}

status RemoveList(LISTS& Lists, char ListName[])
// Lists中删除一个名称为ListName的线性链表
{
	for (int i = 0; i < Lists.length; i++)
	{
		int state = 1;
		for (int j = 0; ListName[j] != '\0'; j++)
		{
			if (Lists.elem[i].name[j] != ListName[j])
			{
				state = 0;
				break;
			}
		}
		if (state)
		{
			DestroyList(Lists.elem[i].L);
			for (int k = i; k < Lists.length - 1; k++)
			{
				Lists.elem[k] = Lists.elem[k + 1];
			}
			Lists.length--;
			return OK;
		}
	}
	return ERROR;
}

int LocateList(LISTS Lists, char ListName[])
// 在Lists中查找一个名称为ListName的线性链表，成功返回逻辑序号，否则返回0
{
	for (int i = 0; i < Lists.length; i++)
	{
		int state = 1;
		for (int j = 0; ListName[j] != '\0'; j++)
		{
			if (Lists.elem[i].name[j] != ListName[j])
			{
				state = 0;
				break;
			}
		}
		if (state)
		{
			return i + 1;
		}
	}
	return ERROR;
}

status MulLists_Input(LISTS& Lists)
//输入数据至多线性链表中
{
	int state;
	int i = 0;
	printf("是否输入数据？（1：继续输入，-1：结束输入）\n");
	while (scanf("%d", &state) == 1 && state == 1)
	{
		if (Lists.length >= 10)
		{
			printf("多线性链表已满，不能再继续输入数据！");
			return OVERFLOW;
		}
		printf("请输入线性链表名称：\n");
		scanf("%s", Lists.elem[i].name);
		InitList(Lists.elem[i].L);
		ListInput(Lists.elem[i].L);
		i++;
		Lists.length++;
		printf("是否继续输入数据？（1：继续输入，-1：结束输入）\n");
	}
	return OK;
}

status Clear_or_Ini_Lists(LISTS& Lists)
//将多线性链表清空或初始化
{
	for (int i = 0; i < Lists.length; i++)
		ClearList(Lists.elem[i].L);
	Lists.length = 0;
	return OK;
}

status ListsEmpty(LISTS Lists)
//判断多线性链表是否为空
{
	if (Lists.length) return FALSE;
	return TRUE;
}

status ListsLength(LISTS Lists)
//返回多线性链表的长度（即线性链表个数）
{
	return Lists.length;
}

status GetList(LISTS Lists, int i)
//获取多线性链表中第i个线性链表的名称及数据
{
	if (i<0 || i>Lists.length)
	{
		printf("输入的位置不合理！\n");
		return INFEASIBLE;
	}
	printf("第%d号线性链表是：\n", i);
	printf("%s ", Lists.elem[i - 1].name);
	ListTraverse(Lists.elem[i - 1].L);
	printf("\n");
	return OK;
}

status PriorList(LISTS Lists, char ListName[])
//获取名称为Listname的线性链表的前驱线性链表
{
	for (int i = 0; i < Lists.length - 1; i++)
	{
		if (strcmp(Lists.elem[i + 1].name, ListName) == 0)
		{
			printf("线性链表%s的前驱是：\n", ListName);
			printf("%s ", Lists.elem[i].name);
			ListTraverse(Lists.elem[i].L);
			printf("\n");
			return OK;
		}
	}
	printf("线性链表%s无前驱！\n", ListName);
	return ERROR;
}

status NextList(LISTS Lists, char ListName[])
//获取名称为Listname的线性链表的后继线性链表
{
	for (int i = 0; i < Lists.length - 1; i++)
	{
		if (strcmp(Lists.elem[i].name, ListName) == 0)
		{
			printf("线性链表%s的后继是：\n", ListName);
			printf("%s ", Lists.elem[i + 1].name);
			ListTraverse(Lists.elem[i + 1].L);
			printf("\n");
			return OK;
		}
	}
	printf("线性链表%s无后继！\n", ListName);
	return ERROR;
}

status ListInsert1(LISTS& Lists, int i, LinkList L1, char ListName[])
//在多线性链表第i个线性链表前插入一个名称为Listname的线性链表并输入其数据
{
	if (Lists.length >= 10)
	{
		printf("多线性链表已满，不能执行插入操作！");
		return OVERFLOW;
	}
	if (i<1 || i>Lists.length + 1) return ERROR;
	if (Lists.length >= 10) return INFEASIBLE;
	for (int j = Lists.length; j >= i; j--)
		Lists.elem[j] = Lists.elem[j - 1];
	strcpy(Lists.elem[i - 1].name, ListName);
	Lists.elem[i - 1].L = L1;
	Lists.length++;
	return OK;
}

status ListsTraverse(LISTS Lists)
//遍历多线性链表，打印所有线性链表的信息
{
	printf("----------------data------------------\n");
	for (int i = 0; i < Lists.length; i++)
	{
		printf("%s    ", Lists.elem[i].name);
		ListTraverse(Lists.elem[i].L);
		printf("\n");
	}
	printf("--------------------------------------\n");
	return OK;
}

void Operation_For_Single_List(void)
//进入对多线性链表的各子线性链表进行操作的系统
{
	int op = 1;
	ElemType i, e, pre, next, x;
	char Lname1[30];
	while (op) {
		system("cls");
		printf("\n\n");
		printf("   Menu for Child_Linear Table On Sequence Structure \n");
		printf("--------------------------------------------------------\n");
		printf("    	  1. InitList         8. PriorElem\n");
		printf("    	  2. DestroyList      9. NextElem\n");
		printf("    	  3. ClearList       10. ListInsert\n");
		printf("    	  4. ListEmpty       11. ListDelete\n");
		printf("    	  5. ListLength      12. ListTraverse\n");
		printf("    	  6. GetElem         13. ListInput\n");
		printf("          7. LocateElem\n");
		printf("    	  0. Exit\n");
		printf("--------------------------------------------------------\n");
		printf("    请选择你的操作[0~13]:\n");
		scanf("%d", &op);
		if (op == 0) break;
		printf("请选择要进行此操作的子线性链表的名称：");
		scanf("%s", Lname1);
		int k;
		for (k = 0; k < Lists.length; k++)
			if (strcmp(Lists.elem[k].name, Lname1) == 0) break;
		if (k == Lists.length)
		{
			printf("不存在此线性链表！\n");
			break;
		}
		switch (op) {
		case 1:
			x = InitList(Lists.elem[k].L);
			if (x == OK) printf("线性链表%s创建成功！\n", Lname1);
			else printf("线性链表%s已存在，创建失败！\n", Lname1);
			getchar(); getchar();
			break;
		case 2:
			if (DestroyList(Lists.elem[k].L) == OK) printf("线性链表%s销毁成功！\n", Lname1);
			else printf("线性链表%s不存在，销毁失败！\n", Lname1);
			getchar(); getchar();
			break;
		case 3:
			if (ClearList(Lists.elem[k].L) == OK) printf("线性链表%s清空成功！\n", Lname1);
			else printf("线性链表%s不存在，清空失败！\n", Lname1);
			getchar(); getchar();
			break;
		case 4:
			x = ListEmpty(Lists.elem[k].L);
			if (x == TRUE) printf("线性链表%s为空！\n", Lname1);
			else if (x == FALSE) printf("线性链表%s非空！\n", Lname1);
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 5:
			if (ListLength(Lists.elem[k].L) != INFEASIBLE)
				printf("线性链表%s长度为%d！\n", Lname1, ListLength(Lists.elem[k].L));
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 6:
			printf("请输入要取的元素的序号：\n");
			scanf("%d", &i);
			x = GetElem(Lists.elem[k].L, i, e);
			if (x == OK) printf("第%d号元素是%d！\n", i, e);
			else if (x == ERROR) printf("线性链表%s中不存在第%d号元素！\n", Lname1, i);
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 7:
			printf("请输入要定位的元素：\n");
			scanf("%d", &e);
			x = LocateElem(Lists.elem[k].L, e);
			if (x == ERROR) printf("线性链表%s中无元素%d！\n", Lname1, e);
			else if (x == INFEASIBLE) printf("线性链表%s不存在！\n", Lname1);
			else printf("元素%d第一次出现在第%d位上！\n", e, x);
			getchar(); getchar();
			break;
		case 8:
			printf("请输入要查找前驱的元素：\n");
			scanf("%d", &e);
			x = PriorElem(Lists.elem[k].L, e, pre);
			if (x == OK) printf("元素%d的前驱是%d！\n", e, pre);
			else if (x == ERROR) printf("元素%d无前驱！\n", e);
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 9:
			printf("请输入要查找后继的元素：\n");
			scanf("%d", &e);
			x = NextElem(Lists.elem[k].L, e, next);
			if (x == OK) printf("元素%d的后继是%d!\n", e, next);
			else if (x == ERROR) printf("元素%d无后继！\n", e);
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 10:
			printf("请输入要插入的元素：\n");
			scanf("%d", &e);
			printf("请输入要插入的位置：\n");
			scanf("%d", &i);
			x = ListInsert(Lists.elem[k].L, i, e);
			if (x == OK) printf("元素插入成功！\n");
			else if (x == ERROR) printf("元素插入失败！\n");
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 11:
			printf("请输入要删除的元素的位置：\n");
			scanf("%d", &i);
			x = ListDelete(Lists.elem[k].L, i, e);
			if (x == OK) printf("第%d号元素删除成功，其值为%d！\n", i, e);
			else if (x == ERROR) printf("元素删除失败！\n");
			else printf("线性链表%s不存在！\n", Lname1);
			getchar(); getchar();
			break;
		case 12:
			if (!ListTraverse(Lists.elem[k].L)) printf("线性链表是空表！\n");
			getchar(); getchar();
			break;
		case 13:
			if (ListInput(Lists.elem[k].L) == OK) printf("线性链表数据输入成功！\n");
			else printf("线性链表数据输入失败！\n");
			getchar(); getchar();
			break;
		case 0:
			break;
		default:
			printf("无此操作，请重新选择操作数！\n");
			break;
		}//end of switch
	}//end of while
	printf("退出对单个子线性链表的操作！\n");
	getchar();
}

status SaveList(LinkList L, char FileName[])
// 如果线性链表L存在，将线性链表L的的元素写到FileName文件中，返回OK，否则返回INFEASIBLE。
{
	LNode* p;
	if (L)
	{
		FILE* fp;
		if ((fp = fopen(FileName, "w")) == NULL) return ERROR;
		p = L->next;
		while (p)
		{
			fprintf(fp, "%d ", p->data);
			p = p->next;
		}
		fclose(fp);
		return OK;
	}
	return INFEASIBLE;
}

status LoadList(LinkList& L, char FileName[])
// 如果线性链表L不存在，将FileName文件中的数据读入到线性链表L中，返回OK，否则返回INFEASIBLE。
{
	LNode* p;
	ClearList(L);
	L = (LinkList)malloc(sizeof(LNode));
	L->next = NULL;
	p = L;
	FILE* fp;
	if ((fp = fopen(FileName, "r")) == NULL) return ERROR;
	int x;
	while (fscanf(fp, "%d", &x) != EOF && x != -1)
	{
		p->next = (LinkList)malloc(sizeof(LNode));
		p = p->next;
		p->data = x;
	}
	p->next = NULL;
	fclose(fp);
	return OK;
}

void File_Reading_And_Writing(void)
//进入线性链表的的文件读写系统
{
	int tp = 1, p = 1;
	int t;
	char Lname2[30];
	while (tp) {
		system("cls");	printf("\n\n");
		printf("   Menu for File Save&Load for Linear List On Sequence Structure \n");
		printf("---------------------------------------------------------------------\n");
		printf("    	        1. SaveList      2. LoadList\n");
		printf("    	        0. Exit\n");
		printf("---------------------------------------------------------------------\n");
		printf("    请选择你的操作[0~2]:\n");
		scanf("%d", &tp);
		if (tp == 0) break;
		printf("请输入要执行此操作的线性表的名称：\n");
		scanf("%s", Lname2);
		int k;
		for (k = 0; k < Lists.length; k++)
			if (strcmp(Lists.elem[k].name, Lname2) == 0) break;
		if (k == Lists.length)
		{
			printf("不存在此线性表，请重新输入！\n");
			break;
		}
		char file[30];
		switch (tp)
		{
		case 1:
			printf("请输入文件名称(以.txt结尾)：\n");
			scanf("%s", file);
			if (!SaveList(Lists.elem[k].L, file)) printf("File open error!\n");
			else printf("保存成功!\n");
			getchar(); getchar();
			break;
		case 2:
			if (!ListEmpty(Lists.elem[k].L))
			{
				printf("线性表非空，读入数据会覆盖原数据造成原始数据丢失，是否继续（1：继续，-1：退出操作）!\n");
				scanf("%d", &p);
			}
			if (p == -1) break;
			printf("请输入文件名称(以.txt或.dat结尾)：\n");
			scanf("%s", file);
			t = LoadList(Lists.elem[k].L, file);
			if (!t) printf("File open error!\n");
			else if (t == OK) printf("读取成功!\n");
			getchar(); getchar();
			break;
		case 0:
			break;
		default:
			printf("无此操作，请重新选择操作数！\n");
			break;
		}
	}
	printf("已退出文件读写系统！\n");
}
*/


//3.OperationOnBiTree(二叉树操作小程序)
/*#define _CRT_SECURE_NO_WARNINGS
#define gets gets_s


#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
typedef int status;
typedef int KeyType;
typedef struct
{
	KeyType  key;
	char others[20];
} TElemType; //二叉树结点类型定义

typedef struct BiTNode
{    //二叉链表结点的定义
	TElemType data;
	struct BiTNode* lchild, * rchild;
} BiTNode, * BiTree;
#define LENG sizeof(struct BiTNode)
#define MaxSize 100
//这样可以通过说明语句：BiTree T;   定义根结点指针T，使得T可以表示一棵二叉树

typedef struct
{
	struct
	{
		char name[30];
		BiTree T;
	} elem[10];
	int length;
}Forest;
Forest forest;

typedef struct QueNode   //树的队列结构体
{
	BiTree p;
	struct QueNode* next;
}QueNode;

int i = 0;//全局变量i，保证CreateBiTree()函数功能
BiTree T = NULL;
BiTree* address;//定位多二叉树中单个二叉树

status CreateBiTree(BiTree& T, TElemType definition[]);//根据带空子树的二叉树先序遍历序列definition构造一个二叉树T
void Visit(BiTree T);//访问树结点函数
void PreOrderTraverse(BiTree T);
void InOrderTraverse(BiTree T);
void PostOrderTraverse(BiTree T);
void LevelOrderTraverse(BiTree T);//层序遍历
status BiTreeEmpty(BiTree T);//二叉树判空
status ClearBiTree(BiTree& T);//将二叉树设置成空，并删除所有结点，释放结点空间
int BiTreeDepth(BiTree T);//求二叉树T的深度
BiTNode* LocateNode(BiTree T, KeyType e);//查找结点
status Assign(BiTree& T, KeyType e, TElemType value);//查找结点关键字等于e的结点，将结点值修改成value
BiTNode* GetParent(BiTree T, KeyType e);
BiTNode* GetSibling(BiTree T, KeyType e);//查找结点关键字等于e的结点的兄弟结点，返回其兄弟结点指针
status InsertNode(BiTree& T, KeyType e, int LR, TElemType c);//插入结点
status DeleteNode(BiTree& T, KeyType e); //删除结点
status SaveBiTree(BiTree T, char FileName[]); //将二叉树的结点数据写入到文件FileName中
status LoadBiTree(BiTree& T, char FileName[]);//读入文件FileName的结点数据，创建二叉树
status MultipleBiTreesInput(Forest& forest);//多二叉树输入
status AddBiTree(Forest& forest, char BiTreeName[]);//在forest中增加一个名称为BiTreeName的空二叉树
status RemoveBiTree(Forest& forest, char BiTreeName[]);///在forest中删除一个名称为BiTreeName的二叉树
status InsertBiTree(Forest& forest, int i, char BiTreeName[]);//在多二叉树第i个二叉树前插入一个名称为BiTreeName的二叉树并输入其数据
status ShowForest(Forest& forest);//遍历打印森林
status ClearForest_or_InitForest(Forest& forest);//将多二叉树清空或初始化




int main(void)
{
	int op = 1, tp = 0;   //操作指令
	int num = 0;   //definition[]数组长度
	int t = 0;   //用于存储函数调用状态信息
	int target = 0;//用于定位判断
	KeyType e = 0;   //关键字变量，用于查找函数
	TElemType NewT;   //用于更新结点的信息暂存
	BiTree find = NULL;
	while (op)
	{
		system("cls");	printf("\n\n");
		printf("      Menu for BiTree On Node Structure \n");
		printf("-------------------------------------------------\n");
		printf("    	  1. CreateBiTree          2. DestroyBiTree\n");
		printf("    	  3. ClearBiTree           4. BiTreeEmpty\n");
		printf("    	  5. BiTreeDepth           6. LocateNode\n");
		printf("    	  7. Assign                8. GetSibling\n");
		printf("    	  9. InsertNode            10. DeleteNode\n");
		printf("    	  11. PreOrderTraverse     12. InOrderTraverse\n");
		printf("    	  13. PostOrderTraverse    14. LevelOrderTraverse\n");
		printf("    	  15. Forest Mode          16. FileSL Mode\n");
		printf("    	  0. Exit\n");
		printf("-------------------------------------------------\n");
		printf("    请选择你的操作[0~16]:");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			if (T)
			{
				printf("二叉树已创建，创建失败！\n");
				getchar(); getchar();
				break;
			}
			printf("请输入二叉树的前序遍历序列，输入关键字和其他信息，用空格隔开：\n");
			printf("请包括空子树(0 null)和结尾标志(-1 null)\n");
			TElemType definition[101];
			num = 0;
			i = 0;
			scanf("%d%s", &definition[0].key, definition[0].others);
			while (definition[num].key + 1)
			{
				num++;
				scanf("%d%s", &definition[num].key, definition[num].others);
			}
			//for (int i = 0; i <= num; i++)
				//printf("%d,%s  ", definition[i].key, definition[i].others);
			t = CreateBiTree(T, definition);
			*address = T;
			if (t) printf("二叉树创建成功！\n");
			else printf("关键字不唯一,创建失败，请重试！\n");
			getchar(); getchar();
			break;
		case 2:
			ClearBiTree(T);
			*address = T;
			printf("二叉树销毁成功！\n");
			getchar(); getchar();
			break;
		case 3:
			ClearBiTree(T);
			*address = T;
			printf("二叉树清空成功！\n");
			getchar(); getchar();
			break;
		case 4:
			t = BiTreeEmpty(T);
			if (t) printf("该二叉树为空！\n");
			else printf("该二叉树非空！\n");
			getchar(); getchar();
			break;
		case 5:
			t = BiTreeDepth(T);
			printf("该二叉树深度为%d\n", t);
			getchar(); getchar();
			break;
		case 6:
			printf("请输入要查找的关键字:\n");
			scanf("%d", &e);
			find = LocateNode(T, e);
			if (find)
			{
				printf("查找成功！该关键字在二叉树中的信息为：\n");
				Visit(find);
			}
			else printf("查找失败！\n");
			getchar(); getchar();
			break;
		case 7:
			printf("请输入要赋值的结点关键字，新的关键字和结点信息：\n");
			scanf("%d%d%s", &e, &NewT.key, NewT.others);
			t = Assign(T, e, NewT);
			*address = T;
			if (t)
			{
				printf("赋值成功！更新后的二叉树前序序列为：\n");
				PreOrderTraverse(T);
				putchar('\n');
				printf("赋值成功！更新后的二叉树中序序列为：\n");
				InOrderTraverse(T);
			}
			else printf("赋值失败！关键字不存在或赋值后不唯一，请重试！\n");
			getchar(); getchar();
			break;
		case 8:
			printf("请输入要查找兄弟结点的关键字:\n");
			scanf("%d", &e);
			find = GetSibling(T, e);
			if (find)
			{
				printf("查找成功！该关键字在二叉树中的兄弟结点为：\n");
				Visit(find);
			}
			else printf("无兄弟结点！\n");
			getchar(); getchar();
			break;
		case 9:
			printf("请输入插入位置的结点关键字，插入方式，待插入关键字和信息：\n");
			printf("插入方式：0=作为左孩子，1=作为右孩子，-1=作为根节点\n");
			int LR;
			scanf("%d%d%d%s", &e, &LR, &NewT.key, NewT.others);
			t = InsertNode(T, e, LR, NewT);
			if (t)
			{
				printf("插入成功！更新后的二叉树前序序列为：\n");
				PreOrderTraverse(T);
				putchar('\n');
				printf("插入成功！更新后的二叉树中序序列为：\n");
				InOrderTraverse(T);
			}
			else printf("插入失败！关键字不存在或插入后不唯一，请重试！\n");
			*address = T;
			getchar(); getchar();
			break;
		case 10:
			TElemType NewT;
			printf("请输入要删除的结点关键字：\n");
			scanf("%d", &e);
			t = DeleteNode(T, e);
			if (t)
			{
				printf("删除成功！更新后的二叉树前序序列为：\n");
				PreOrderTraverse(T);
				putchar('\n');
				printf("删除成功！更新后的二叉树中序序列为：\n");
				InOrderTraverse(T);
			}
			else printf("删除失败！关键字不存在，请重试！\n");
			*address = T;
			getchar(); getchar();
			break;
		case 11:
			printf("该二叉树前序遍历序列为：\n");
			PreOrderTraverse(T);
			getchar(); getchar();
			break;
		case 12:
			printf("该二叉树中序遍历序列为：\n");
			InOrderTraverse(T);
			getchar(); getchar();
			break;
		case 13:
			printf("该二叉树后序遍历序列为：\n");
			PostOrderTraverse(T);
			getchar(); getchar();
			break;
		case 14:
			printf("该二叉树层序遍历序列为：\n");
			LevelOrderTraverse(T);
			getchar(); getchar();
			break;
		case 15:
		again:
			system("cls");	printf("\n\n");
			printf("欢迎进入多二叉树管理模式!\n");
			printf("      Menu for Multiple BiTrees On Sequence Structure \n");
			printf("-----------------------------------------------------------\n");
			printf("    	  1. MultipleBiTreesInput       2. LocateBiTree\n");
			printf("    	  3. AddBiTree                  4. RemoveBiTree\n");
			printf("    	  5. InsertBiTree               6. ShowForest\n");
			printf("    	  7. ClearForest_or_InitForest      \n");
			printf("    	  0. Exit\n");
			printf("-----------------------------------------------------------\n");
			printf("    请选择你的操作[0~7]:");
			scanf("%d", &tp);
			char s[30];
			switch (tp)
			{
			case 1:
				if (MultipleBiTreesInput(forest) == OK)  printf("输入成功！\n");
				else printf("输入失败！\n");
				getchar(); getchar();
				goto again;
			case 2:
				printf("请输入要进行操作的二叉树名称：\n");
				scanf("%s", s);
				target = 0;
				for (int k = 0; k < forest.length; k++)
				{
					if (!strcmp(forest.elem[k].name, s))
					{
						address = &forest.elem[k].T;
						T = *address;
						target = 1; break;
					}
				}
				if (target) printf("定位成功！\n");
				else printf("定位失败！请重试！\n");
				getchar(); getchar();
				goto again;
			case 3:
				printf("请输入要增加的空二叉树的名称：\n");
				scanf("%s", s);
				if (AddBiTree(forest, s) == OK) printf("空二叉树%s增加成功！\n", s);
				else printf("空二叉树%s增加失败！\n", s);
				getchar(); getchar();
				goto again;
			case 4:
				printf("请输入要删除的空二叉树的名称：\n");
				scanf("%s", s);
				if (RemoveBiTree(forest, s) == OK) printf("空二叉树%s删除成功！\n", s);
				else printf("空二叉树%s删除失败！\n", s);
				getchar(); getchar();
				goto again;
			case 5:
				printf("请输入要插入的二叉树的名称：\n");
				scanf("%s", s);
				printf("请输入要插入的位置：\n");
				scanf("%d", &i);
				if (InsertBiTree(forest, i, s) == OK) printf("二叉树%s插入成功！\n", s);
				else printf("插入位置错误！\n");
				getchar(); getchar();
				goto again;
			case 6:
				if (!ShowForest(forest)) printf("多二叉树为空！");
				getchar(); getchar();
				goto again;
			case 7:
				if (ClearForest_or_InitForest(forest) == OK) printf("多二叉树清空或初始化成功！\n");
				else printf("多二叉树清空或初始化失败！\n");
				getchar(); getchar();
				break;
			case 0:
				printf("已退出多二叉树管理模式!"); break;
			default:
				printf("操作有误，请重试!"); getchar(); getchar(); goto again;
			}
			getchar(); getchar();
			break;
		case 16:
		oncemore:
			system("cls");	printf("\n\n");
			printf("欢迎进入文件读写模式!\n");
			printf("   Menu for File Save&Load for BiTree On Sequence Structure \n");
			printf("---------------------------------------------------------------------\n");
			printf("    	        1. SaveList        2. LoadList\n");
			printf("    	        0. Exit\n");
			printf("---------------------------------------------------------------------\n");
			printf("    请选择你的操作[0~2]:");
			scanf("%d", &tp);
			char file[30];
			switch (tp)
			{
			case 1:
				printf("请输入文件名称(以.dat或.txt结尾)：\n");
				scanf("%s", file);
				if (!SaveBiTree(T, file)) printf("File open error!\n");
				else printf("保存成功!\n");
				getchar(); getchar();
				goto oncemore;
			case 2:
				printf("请输入文件名称(以.dat或.txt结尾)：\n");
				getchar();
				gets(file);
				i = 0;
				t = LoadBiTree(T, file);
				*address = T;
				if (!t) printf("File open error!\n");
				else if (t == OK) printf("读取成功!\n");
				else printf("二叉树非空，读入数据会覆盖原数据造成数据丢失，请重试!\n");
				getchar();
				goto oncemore;
			case 0:
				printf("已退出文件读写模式!"); break;
			default:
				printf("操作有误，请重试!"); getchar(); getchar(); goto oncemore;
			}
			getchar(); getchar();
		case 0:
			break;
		default:printf("操作有误，请重试!");
			getchar(); getchar();
		}
	}
	printf("欢迎下次再使用本系统！\n");
	return 0;
}




status CreateBiTree(BiTree& T, TElemType definition[])
{
	if (i == 0)
	{
		int len = 0;
		while (definition[len].key + 1) len++;
		for (int j = 0; j < len - 1; j++)
		{
			if (!definition[j].key) continue;
			for (int k = j + 1; k < len; k++)
			{
				if (!definition[k].key) continue;
				if (definition[j].key == definition[k].key) return ERROR;
			}
		}
	}
	if (definition[i].key == 0)
	{
		T = NULL;
		i++;
		return OK;
	}
	else if (definition[i].key + 1)
	{
		T = (BiTree)malloc(LENG);
		T->data = definition[i];
		i++;
		return CreateBiTree(T->lchild, definition) && CreateBiTree(T->rchild, definition);
	}
	return OK;
}

void Visit(BiTree T)
{
	printf("%d,%s ", T->data.key, T->data.others);
}

void PreOrderTraverse(BiTree T)
{
	if (!T) return;
	BiTree Stack[MaxSize], p;   //定义一个栈
	int top = -1;   //初始化栈
	Stack[++top] = T;   //根结点入栈
	while (top != -1)   //栈空循环退出，遍历结束
	{
		p = Stack[top--];   //出栈并输出栈顶结点
		Visit(p);   //Visit()为访问p的函数
		if (p->rchild)   //栈顶结点的右孩子存在，则右孩子入栈
			Stack[++top] = p->rchild;
		if (p->lchild)   //栈顶结点的左孩子存在，则左孩子入栈
			Stack[++top] = p->lchild;
	}
}

void InOrderTraverse(BiTree T)
{
	if (!T) return;
	BiTree Stack[MaxSize], p = T;   //定义一个栈
	int top = -1;   //初始化栈
	//下面这个循环完成中序遍历，注意：进栈、出栈过程可能出现栈空状态
	//但此时遍历还未结束，因根结点的右子树还没有遍历，此时p非空，根据这一点维持循环的进行
	while (top != -1 || p)
	{
		while (p)   //左孩子存在，则左孩子入栈
		{
			Stack[++top] = p;
			p = p->lchild;
		}
		if (top != -1)   //在栈非空的情况下，出栈并输出出栈结点
		{
			p = Stack[top--];
			Visit(p);
			p = p->rchild;
		}
	}
}

void PostOrderTraverse(BiTree T)
{
	if (!T) return;
	BiTree Stack1[MaxSize], Stack2[MaxSize], p = NULL;   //定义两个栈
	int top1 = -1, top2 = -1;
	Stack1[++top1] = T;
	while (top1 != -1)
	{
		p = Stack1[top1--];
		Stack2[++top2] = p;   //注意这里和先序遍历的区别，输出改为入Stack2
		//注意下边这两个if语句和先序遍历的区别，左、右孩子的入栈顺序相反
		if (p->lchild)
			Stack1[++top1] = p->lchild;
		if (p->rchild)
			Stack1[++top1] = p->rchild;
	}
	while (top2 != -1)
	{
		//出栈序列即为后序遍历序列
		p = Stack2[top2--];
		Visit(p);
	}
}

void LevelOrderTraverse(BiTree T)
{
	if (!T) return;
	int front, rear;
	BiTree queue[MaxSize];   //定义一个循环队列，用来记录将要访问的层次上的结点
	front = rear = 0;
	BiTree q;
	rear = (rear + 1) % MaxSize;
	queue[rear] = T;   //根节点入队
	while (front != rear)   //当队列不为空时进入循环
	{
		front = (front + 1) % MaxSize;
		q = queue[front];   //队首结点出队
		Visit(q);   //访问队首结点
		if (q->lchild)   //左子树根节点入队
		{
			rear = (rear + 1) % MaxSize;
			queue[rear] = q->lchild;
		}
		if (q->rchild)   //右子树根节点入队
		{
			rear = (rear + 1) % MaxSize;
			queue[rear] = q->rchild;
		}
	}
}

status BiTreeEmpty(BiTree T)
{
	if (!T) return TRUE;
	return FALSE;
}

status ClearBiTree(BiTree& T)
{
	if (!T) return OK;
	if (T->lchild)
		ClearBiTree(T->lchild);
	if (T->rchild)
		ClearBiTree(T->rchild);
	free(T);
	T = NULL;
	return OK;
}

int BiTreeDepth(BiTree T)
{
	int ld, rd;
	if (T == NULL)
		return 0;   //空树则深度为0
	ld = BiTreeDepth(T->lchild);   //求左子树深度
	rd = BiTreeDepth(T->rchild);   //求右子树深度
	return (ld > rd ? ld : rd) + 1;   //返回左、右子树深度的最大值加1，即为整棵树深度
}

BiTNode* LocateNode(BiTree T, KeyType e)
{
	BiTree q = NULL;
	if (!T) return NULL;
	if (T->data.key == e) q = T;
	else q = LocateNode(T->lchild, e);
	if (!q) q = LocateNode(T->rchild, e);
	return q;
}

status Assign(BiTree& T, KeyType e, TElemType value)
{
	if (e != value.key && LocateNode(T, value.key)) return ERROR;
	BiTree q = LocateNode(T, e);
	if (q)
	{
		q->data = value;
		return OK;
	}
	return ERROR;
}

BiTNode* GetParent(BiTree T, KeyType e)
{
	BiTree q = NULL;
	if (!T) return NULL;
	if (T->data.key == e) return NULL;
	if (T->lchild)
		if (T->lchild->data.key == e) q = T;
	if (T->rchild)
		if (T->rchild->data.key == e) q = T;
	if (!q) q = GetParent(T->lchild, e);
	if (!q) q = GetParent(T->rchild, e);
	return q;
}

BiTNode* GetSibling(BiTree T, KeyType e)
{
	BiTree q = LocateNode(T, e);
	if (q)
	{
		BiTree p = GetParent(T, e);
		if (p->lchild && p->lchild->data.key == e) return p->rchild;
		else return p->lchild;
	}
	return q;
}

status InsertNode(BiTree& T, KeyType e, int LR, TElemType c)
{
	//LR为0或1，c是待插入结点；根据LR为0或者1，插入结点c到T中，作为关键字为e的结点的左或右孩子结点
	//结点e的原有左子树或右子树则为结点c的右子树，返回OK。如果插入失败，返回ERROR。
	//特别地，当LR为-1时，作为根结点插入，原根结点作为c的右子树。
	if (LR == -1)
	{
		BiTree NewT = (BiTree)malloc(LENG);
		NewT->data = c;
		NewT->rchild = T;
		NewT->lchild = NULL;
		T = NewT;
		return OK;
	}
	if (LocateNode(T, c.key)) return ERROR;
	BiTree q = LocateNode(T, e);
	if (q)
	{
		BiTree NewT = (BiTree)malloc(LENG), p = NULL;
		NewT->data = c;
		if (LR == 0)
		{
			p = q->lchild;
			q->lchild = NewT;
		}
		else if (LR == 1)
		{
			p = q->rchild;
			q->rchild = NewT;
		}
		NewT->rchild = p;
		NewT->lchild = NULL;
		return OK;
	}
	return ERROR;
}

status DeleteNode(BiTree& T, KeyType e)
{
	//删除T中关键字为e的结点；同时，如果关键字为e的结点度为0，删除即可
	//如关键字为e的结点度为1，用关键字为e的结点孩子代替被删除的e位置
	//如关键字为e的结点度为2，用e的左孩子代替被删除的e位置，e的右子树作为e的左子树中最右结点的右子树
	BiTree q = LocateNode(T, e);
	if (q)
	{
		int degree = 0, lt = 0, rt = 0;
		if (q->lchild) { degree++; lt = 1; }
		if (q->rchild) { degree++; rt = 1; }
		BiTree p = GetParent(T, e);
		if (degree == 0)
		{
			if (!p)
			{
				free(q);
				T = NULL;
				return OK;
			}
			else if (p->lchild && p->lchild->data.key == e)
			{
				p->lchild = NULL;
				free(q);
				return OK;
			}
			else
			{
				p->rchild = NULL;
				free(q);
				return OK;
			}
		}
		else if (degree == 1)
		{
			BiTree qchild = NULL;
			if (lt) qchild = q->lchild;
			else qchild = q->rchild;
			if (!p)
			{
				free(q);
				T = qchild;
				return OK;
			}
			else if (p->lchild && p->lchild->data.key == e)
			{
				p->lchild = qchild;
				free(q);
				return OK;
			}
			else
			{
				p->rchild = qchild;
				free(q);
				return OK;
			}
		}
		else
		{
			BiTree qchild = q->lchild, rtPlace = q->lchild;
			while (rtPlace->rchild) rtPlace = rtPlace->rchild;
			rtPlace->rchild = q->rchild;
			if (!p)
			{
				T = qchild;
				free(q);
			}
			else if (p->lchild && p->lchild->data.key == e)
			{
				p->lchild = qchild;
				free(q);
			}
			else
			{
				p->rchild = qchild;
				free(q);
			}
			return OK;
		}
	}
	return ERROR;
}

status SaveBiTree(BiTree T, char FileName[])
{
	if (!T) return INFEASIBLE;   //未初始化的空树不能进行写入操作
	FILE* fp;
	if ((fp = fopen(FileName, "w")) == NULL) return ERROR;
	BiTree Stack[MaxSize], p;   //定义一个栈
	int top = -1;   //初始化栈
	Stack[++top] = T;   //根节点入栈
	while (top != -1)   //栈空循环退出，遍历结束
	{
		p = Stack[top--];   //出栈并打印栈顶结点
		if (p)
		{
			fprintf(fp, "%d,%s  ", p->data.key, p->data.others);
			Stack[++top] = p->rchild;   //栈顶结点的右孩子入栈
			Stack[++top] = p->lchild;   //栈顶结点的左孩子入栈
		}
		else fprintf(fp, "0,null  ");
	}
	fprintf(fp, "-1,null  ");
	fclose(fp);
	return OK;
}

status LoadBiTree(BiTree& T, char FileName[])
{
	if (T)
	{
		printf("二叉树T中已经有数据，读入数据会覆盖原数据造成数据丢失！\n");
		return INFEASIBLE;
	}
	TElemType definition[MaxSize];
	FILE* fp;
	if ((fp = fopen(FileName, "r")) == NULL) return ERROR;
	fscanf(fp, "%d,%s", &definition[0].key, definition[0].others);
	int num = 0;
	while (definition[num].key + 1)
	{
		num++;
		fscanf(fp, "%d,%s", &definition[num].key, definition[num].others);
	}
	if (CreateBiTree(T, definition)) return OK;
}

status MultipleBiTreesInput(Forest& forest)
{
	int state;
	int j = 0, num = 0;
	printf("输入多个二叉树及其内部数据：\n");
	printf("输入1开始！\n");
	TElemType definition[MaxSize];
	while (scanf("%d", &state) == 1 && state == 1)
	{
		num = 0;
		if (forest.length >= 10)
		{
			printf("多二叉树已满，不能再继续输入数据！");
			return OVERFLOW;
		}
		printf("请输入二叉树名称：\n");
		scanf("%s", forest.elem[j].name);
		ClearBiTree(forest.elem[j].T);
		printf("请输入二叉树的前序遍历序列，输入关键字和其他信息，用空格隔开：\n");
		printf("请包括空子树(0 null)和结尾标志(-1 null)\n");
		scanf("%d%s", &definition[0].key, definition[0].others);
		while (definition[num].key + 1)
		{
			num++;
			scanf("%d%s", &definition[num].key, definition[num].others);
		}
		i = 0;
		CreateBiTree(forest.elem[j].T, definition);
		j++;
		forest.length++;
		printf("是否继续？1：继续输入；-1：结束输入！\n");
	}
	return OK;
}

status AddBiTree(Forest& forest, char BiTreeName[])
{
	ClearBiTree(forest.elem[forest.length].T);
	int i;
	strcpy(forest.elem[forest.length].name, BiTreeName);
	forest.length++;
	return OK;
}

int LocateT(Forest forest, char Tname[])  //寻找对应的树，返回对应线性表的位置下标
{
	int i, j;
	for (i = 0; i < forest.length; i++)//遍历多个名称的线性表
	{
		for (j = 0; Tname[j] != 0 && forest.elem[i].name[j] != 0; j++)//遍历各线性表的名称字符串
		{
			if (forest.elem[i].name[j] != Tname[j])
				break;
		}
		if (Tname[j] == 0 && forest.elem[i].name[j] == 0)
			return i + 1;
	}
	return 0;
}

status RemoveBiTree(Forest& Bitrees, char Tname[])// Bitrees中删除一个名称为Tname的线性表,成功返回OK，否则ERROR
{
	int i = LocateT(Bitrees, Tname);
	BiTree tem = Bitrees.elem[i - 1].T;
	for (int j = i - 1; j <= Bitrees.length - 2; j++)
	{
		Bitrees.elem[j] = Bitrees.elem[j + 1];
	}
	ClearBiTree(tem);
	Bitrees.length--;
	return OK;
}

status InsertBiTree(Forest& forest, int i, char BiTreeName[])
{//在多二叉树的第i个二叉树前插入一个名称为Listname的二叉树并输入其数据	
	if (forest.length >= 10)
	{
		printf("多二叉树已满，不能执行插入操作！");
		return OVERFLOW;
	}
	if (i<1 || i>forest.length + 1) return ERROR;
	if (forest.length >= 10) return INFEASIBLE;
	TElemType definition[40];
	for (int j = forest.length; j >= i; j--)
		forest.elem[j] = forest.elem[j - 1];
	strcpy(forest.elem[i - 1].name, BiTreeName);
	int num = 0;
	scanf("%d%s", &definition[0].key, definition[0].others);
	while (definition[num].key + 1)
	{
		num++;
		scanf("%d%s", &definition[num].key, definition[num].others);
	}
	BiTree T = NULL;
	CreateBiTree(T, definition);
	forest.elem[i - 1].T = T;
	forest.length++;
	return OK;
}

status ShowForest(Forest& forest)
{
	if (!forest.length) return ERROR;
	int i;
	for (i = 0; i < forest.length; i++)
	{
		printf("-------------------------------data---------------------------------\n");
		printf("名称：%s\n", forest.elem[i].name);
		printf("前序： "); PreOrderTraverse(forest.elem[i].T);
		putchar('\n');
		printf("中序： "); InOrderTraverse(forest.elem[i].T);
		putchar('\n');
		printf("-------------------------------end----------------------------------\n");
	}
	return OK;
}

status ClearForest_or_InitForest(Forest& forest)
{
	for (int i = 0; i < forest.length; i++)
		ClearBiTree(forest.elem[i].T);
	forest.length = 0;
	return OK;
}
*/


//4.OperationOnGraph(图操作小程序)
/*
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
#define MAX_VERTEX_NUM 20
#define maxSize 100
int IsVisited[MAX_VERTEX_NUM];
typedef int status;
typedef int KeyType;
typedef enum { DG, DN, UDG, UDN } GraphKind;
typedef struct
{
	KeyType  key;
	char others[20];
} VertexType; //顶点类型定义

typedef struct ArcNode
{         //表结点类型定义
	int adjvex;              //顶点位置编号 
	struct ArcNode* nextarc;       //下一个表结点指针
} ArcNode;

typedef struct VNode
{                //头结点及其数组类型定义
	VertexType data;           //顶点信息
	ArcNode* firstarc;           //指向第一条弧
} VNode, AdjList[MAX_VERTEX_NUM];

typedef  struct
{  //邻接表的类型定义
	AdjList vertices;          //头结点数组
	int vexnum, arcnum;         //顶点数、弧数
	GraphKind  kind;        //图的类型
} ALGraph;
//这样可以通过说明语句：ALGraph G;   定义结构变量G，使得G可以表示一个图。

typedef struct
{
	struct
	{
		char name[30];
		ALGraph G;
	} elem[10];
	int length;
}MultipleGraphs;

ALGraph G;
MultipleGraphs Gs;
ALGraph* address = NULL;   //定位多图中单个图
VertexType V[30];   //顶点数组
KeyType VR[100][2];   //边数组

void ShowGraph(ALGraph G);//打印邻接表
status CreateGraph(ALGraph& G, VertexType V[], KeyType VR[][2]);//根据V和VR构造图T
status DestroyGraph(ALGraph& G);//销毁无向图G,删除G的全部顶点和边
int LocateVex(ALGraph G, KeyType u);//查找顶点，查找成功返回位序，否则返回-1
status PutVex(ALGraph& G, KeyType u, VertexType value);//根据u在图G中查找顶点，查找成功将该顶点值修改成value
int FirstAdjVex(ALGraph G, KeyType u);//根据u在图G中查找顶点，查找成功返回顶点u的第一邻接顶点位序
int NextAdjVex(ALGraph G, KeyType v, KeyType w);//根据u在图G中查找顶点，查找成功返回顶点v的邻接顶点相对于w的下一邻接顶点的位序
status InsertVex(ALGraph& G, VertexType v);//在图G中插入顶点v
status DeleteVex(ALGraph& G, KeyType v);//在图G中删除关键字v对应的顶点以及相关的弧
status InsertArc(ALGraph& G, KeyType v, KeyType w);//在图G中增加弧<v,w>
status DeleteArc(ALGraph& G, KeyType v, KeyType w);//在图G中删除弧<v,w>
void visit(VertexType v);//一类访问顶点的函数
status DFSTraverse(ALGraph& G, void (*visit)(VertexType));//对图G进行深度优先搜索遍历
status BFSTraverse(ALGraph& G, void (*visit)(VertexType));//对图G进行广度优先搜索遍历
status SaveGraph(ALGraph G, char FileName[]);//将图的数据写入到文件FileName中
status LoadGraph(ALGraph& G, char FileName[]);//读入文件FileName的图数据，创建图的邻接表
status MultipleGraphsInput(MultipleGraphs& Gs);//多图输入
status RemoveGraph(MultipleGraphs& Gs, char GraphName[]);///在Gs中删除一个名称为GraphName的图
status InsertGraph(MultipleGraphs& Gs, int i, char GraphName[]);//在多图第i个图前插入一个名称为GraphName的图
status ShowMultipleGraphs(MultipleGraphs& Gs);//遍历打印多图
status InitMultipleGraphs(MultipleGraphs& Gs);//初始化多图



int main(void)
{
	int op = 1, tp = 0;   //操作指令
	int t = 0;   //用于存储函数调用状态信息
	int target = 0;   //用于定位判断
	int v, w;   //存储弧的顶点信息
	int i;   //用于数组V，VR构建
	KeyType e = 0;   //关键字变量，用于查找函数
	VertexType NewVex;   //用于更新顶点的信息暂存
	VertexType find;
	while (op)
	{
		system("cls");	printf("\n\n");
		printf("      Menu for ALGraph On Node Structure \n");
		printf("--------------------------------------------------------------\n");
		printf("    	  1. CreateALGraph         2. DestroyALGraph\n");
		printf("    	  3. LocateVex             4. PutVex\n");
		printf("    	  5. FirstAdjVex           6. NextAdjVex\n");
		printf("    	  7. InsertVex             8. DeleteVex\n");
		printf("    	  9. InsertArc             10. DeleteArc\n");
		printf("    	  11. DFSTraverse          12. BFSTraverse\n");
		printf("    	  13. MultipleGraphs Mode  14. FileSL Mode\n");
		printf("    	  0. Exit\n");
		printf("--------------------------------------------------------------\n");
		printf("    请选择你的操作[0~14]:");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			i = 0;
			printf("请输入顶点信息(先关键字再名称，空格隔开,eg.5 线性表)\n");
			do
			{
				scanf("%d%s", &V[i].key, V[i].others);
			} while (V[i++].key != -1);
			i = 0;
			printf("请输入关系对序列,空格隔开\n");
			do
			{
				scanf("%d%d", &VR[i][0], &VR[i][1]);
			} while (VR[i++][0] != -1);
			if (CreateGraph(G, V, VR) == ERROR) printf("输入数据错误，无法创建");
			else
			{
				if (G.arcnum != i - 1) {
					printf("边的数目错误！\n");
					return 0;
				}
				printf("创建成功，该图信息(邻接表)如下:\n");
				ShowGraph(G);
			}
			getchar(); getchar();
			break;
		case 2:
			DestroyGraph(G);
			*address = G;
			printf("图销毁成功！\n");
			getchar(); getchar();
			break;
		case 3:
			printf("请输入要查找的关键字:\n");
			scanf("%d", &e);
			t = LocateVex(G, e);
			if (t + 1)
			{
				printf("查找成功！该关键字在图中的信息为：\n");
				find = G.vertices[t].data;
				visit(find);
			}
			else printf("查找失败！\n");
			getchar(); getchar();
			break;
		case 4:
			printf("请输入要赋值的结点关键字，新的关键字和定点信息：\n");
			scanf("%d%d%s", &e, &NewVex.key, NewVex.others);
			t = PutVex(G, e, NewVex);
			if (t)
			{
				*address = G;
				printf("赋值成功！更新后的邻接表信息如下：\n");
				ShowGraph(G);
			}
			else printf("赋值失败！关键字不存在或赋值后不唯一，请重试！\n");
			getchar(); getchar();
			break;
		case 5:
			printf("请输入要获得第一邻接点的顶点的关键字:\n");
			scanf("%d", &e);
			t = FirstAdjVex(G, e);
			if (t + 1)
			{
				printf("查找成功，该顶点的第一邻接点信息如下:\n");
				visit(G.vertices[t].data);
			}
			else printf("查找失败，不存在该顶点或该顶点没有邻接点！\n");
			getchar(); getchar();
			break;
		case 6:
			printf("请输入要获得下一邻接点的顶点的关键字和该顶点的一个邻接点的关键字:\n");
			int w;
			scanf("%d%d", &e, &w);
			t = NextAdjVex(G, e, w);
			if (t + 1)
			{
				printf("查找成功，该顶点的下一邻接点信息如下:\n");
				visit(G.vertices[t].data);
			}
			else printf("查找失败，不存在该顶点或该顶点没有下一邻接点！\n");
			getchar(); getchar();
			break;
		case 7:
			printf("请输入待插入关键字和信息：\n");
			scanf("%d%s", &NewVex.key, NewVex.others);
			t = InsertVex(G, NewVex);
			if (t)
			{
				printf("插入成功！更新后的邻接表为：\n");
				ShowGraph(G);
			}
			else printf("插入失败！关键字插入后不唯一或图已满，请重试！\n");
			*address = G;
			getchar(); getchar();
			break;
		case 8:
			printf("请输入要删除的顶点关键字：\n");
			scanf("%d", &e);
			t = DeleteVex(G, e);
			if (t)
			{
				printf("删除成功！更新后的邻接表信息为：\n");
				ShowGraph(G);
			}
			else printf("删除失败！关键字不存在或删除后为空图，请重试！\n");
			*address = G;
			getchar(); getchar();
			break;
		case 9:
			printf("请输入要插入的弧<v, w>:\n");
			scanf("%d%d", &v, &w);
			t = InsertArc(G, v, w);
			if (t)
			{
				printf("插入成功！更新后的邻接表为：\n");
				ShowGraph(G);
			}
			else printf("插入失败！顶点不存在或已存在该弧，请重试！\n");
			*address = G;
			getchar(); getchar();
			break;
		case 10:
			printf("请输入要删除的弧<v, w>:\n");
			scanf("%d%d", &v, &w);
			t = DeleteArc(G, v, w);
			if (t)
			{
				printf("删除成功！更新后的邻接表为：\n");
				ShowGraph(G);
			}
			else printf("插入失败！顶点不存在或该弧不存在，请重试！\n");
			*address = G;
			getchar(); getchar();
			break;
		case 11:
			printf("该图的深度优先搜索遍历序列为:\n");
			memset(IsVisited, 0, sizeof(IsVisited));
			DFSTraverse(G, visit);
			getchar(); getchar();
			break;
		case 12:
			printf("该图的广度优先搜索遍历序列为:\n");
			memset(IsVisited, 0, sizeof(IsVisited));
			BFSTraverse(G, visit);
			getchar(); getchar();
			break;
		case 13:
		again:
			system("cls");	printf("\n\n");
			printf("欢迎进入多图管理模式!\n");
			printf("      Menu for Multiple ALGraphs On Sequence Structure \n");
			printf("----------------------------------------------------------------------\n");
			printf("    	  1. MultipleGraphsInput        2. LocateGraph\n");
			printf("    	  3. InitMultipleGraphs         4. RemoveGraph\n");
			printf("    	  5. InsertGraph                6. ShowMultipleGraphs\n");
			printf("    	  0. Exit\n");
			printf("----------------------------------------------------------------------\n");
			printf("    请选择你的操作[0~7]:");
			scanf("%d", &tp);
			char s[30];
			switch (tp)
			{
			case 1:
				if (MultipleGraphsInput(Gs) == OK)  printf("输入成功！\n");
				else printf("输入失败！\n");
				getchar(); getchar();
				goto again;
			case 2:
				printf("请输入要进行操作的图名称：\n");
				scanf("%s", s);
				target = 0;
				for (int k = 0; k < Gs.length; k++)
				{
					if (!strcmp(Gs.elem[k].name, s))
					{
						address = &Gs.elem[k].G;
						G = *address;
						target = 1; break;
					}
				}
				if (target) printf("定位成功！\n");
				else printf("定位失败！请重试！\n");
				getchar(); getchar();
				goto again;
			case 3:
				if (InitMultipleGraphs(Gs) == OK) printf("多图初始化成功！\n");
				else printf("多图初始化失败！\n");
				getchar(); getchar();
				goto again;
			case 4:
				printf("请输入要删除的图的名称：\n");
				scanf("%s", s);
				if (RemoveGraph(Gs, s) == OK) printf("图%s删除成功！\n", s);
				else printf("图%s删除失败！\n", s);
				getchar(); getchar();
				goto again;
			case 5:
				printf("请输入要插入的图的名称：\n");
				scanf("%s", s);
				printf("请输入要插入的位置：\n");
				scanf("%d", &i);
				if (InsertGraph(Gs, i, s) == OK) printf("图%s插入成功！\n", s);
				else printf("插入位置错误！\n");
				getchar(); getchar();
				goto again;
			case 6:
				if (!ShowMultipleGraphs(Gs)) printf("多图为空！");
				getchar(); getchar();
				goto again;
			case 0:
				printf("已退出多图管理模式!"); break;
			default:
				printf("操作有误，请重试!"); getchar(); getchar(); goto again;
			}
			getchar(); getchar();
			break;
		case 14:
		oncemore:
			system("cls");	printf("\n\n");
			printf("欢迎进入文件读写模式!\n");
			printf("   Menu for File Save&Load for BiTree On Sequence Structure \n");
			printf("---------------------------------------------------------------------\n");
			printf("    	        1. SaveGraph        2. LoadGraph\n");
			printf("    	        0. Exit\n");
			printf("---------------------------------------------------------------------\n");
			printf("    请选择你的操作[0~2]:");
			scanf("%d", &tp);
			char file[30];
			switch (tp)
			{
			case 1:
				printf("请输入文件名称(以.dat或.txt结尾)：\n");
				scanf("%s", file);
				if (!SaveGraph(G, file)) printf("File open error!\n");
				else printf("保存成功!\n");
				getchar(); getchar();
				goto oncemore;
			case 2:
				printf("请输入文件名称(以.dat或.txt结尾)：\n");
				getchar();
				gets(file);
				i = 0;
				t = LoadGraph(G, file);
				*address = G;
				if (!t) printf("File open error!\n");
				else if (t == OK) printf("读取成功!\n");
				else printf("图非空，读入数据会覆盖原数据造成数据丢失，请重试!\n");
				getchar(); getchar();
				goto oncemore;
			case 0:
				printf("已退出文件读写模式!"); break;
			default:
				printf("操作有误，请重试!"); 	system("pause"); goto oncemore;
			}
			getchar(); getchar();
		case 0:
			break;
		default:printf("操作有误，请重试!");
			getchar(); getchar();
		}
	}
	printf("欢迎下次再使用本系统！\n");
	return 0;
}


void ShowGraph(ALGraph G)
{
	for (int j = 0; j < G.vexnum; j++)
	{
		ArcNode* p = G.vertices[j].firstarc;
		printf("%d %s", G.vertices[j].data.key, G.vertices[j].data.others);
		while (p)
		{
			printf(" %d", p->adjvex);   //打印邻接边
			p = p->nextarc;
		}
		printf("\n");
	}
	printf("vexnum:%d\n", G.vexnum);//打印顶点数
	printf("arcnum:%d\n", G.arcnum);//打印边数
}

int LocateVertex(VertexType V[], int val)//在V[]中查找val
{
	for (int i = 0; V[i].key + 1; i++)
		if (V[i].key == val) return i;
	return -1;
}

bool CheckV(VertexType V[])//检查V[]中是否有重复关键字，若有则非法
{
	for (int i = 0; V[i + 1].key + 1; i++)
		for (int j = i + 1; V[j].key + 1; j++)
			if (V[i].key == V[j].key) return false;
	return true;
}

bool CheckVR(VertexType V[], KeyType VR[][2])//检查VR[]中是否有不存在的边，若有则非法
{
	for (int i = 0; VR[i][0] + 1; i++)
	{
		if (LocateVertex(V, VR[i][0]) == -1) return false;
		if (LocateVertex(V, VR[i][1]) == -1) return false;
	}
	return true;
}

void SimplifyVR(KeyType VR[][2])//去除重复的边和自身的环
{
	if (VR[0][0] == -1) return;
	int flag = 0;   //指示灯，判断是否执行过删除
	for (int i = 0; VR[i + 1][0] + 1; i++)
	{
		flag = 0;
		for (int j = i + 1; VR[j][0] + 1; j++)
		{
			if ((VR[i][0] == VR[j][0]) && (VR[i][1] == VR[j][1]))
				for (int k = j; VR[k][0] + 1; k++)
				{
					VR[k][0] = VR[k + 1][0];
					VR[k][1] = VR[k + 1][1];
					flag = 1;
				}
			if ((VR[i][1] == VR[j][0]) && (VR[i][0] == VR[j][1]))
				for (int k = j; VR[k][0] + 1; k++)
				{
					VR[k][0] = VR[k + 1][0];
					VR[k][1] = VR[k + 1][1];
					flag = 1;
				}
			if (VR[j][1] == VR[j][0])
				for (int k = j; VR[k][0] + 1; k++)
				{
					VR[k][0] = VR[k + 1][0];
					VR[k][1] = VR[k + 1][1];
				}
			if (flag) j--;   //避免连续多个需要删除时漏删
		}
	}
}

status CreateGraph(ALGraph& G, VertexType V[], KeyType VR[][2])
{
	//根据V和VR构造图T并返回OK，如果V和VR不正确，返回ERROR,如果有相同的关键字，返回ERROR
	G.kind = UDG;
	if (!CheckV(V)) return ERROR;   //检查关键字
	if (!CheckVR(V, VR)) return ERROR;   //检查边
	SimplifyVR(VR);   //简化边
	int i;
	for (i = 0; V[i].key + 1; i++)
	{
		if (i == MAX_VERTEX_NUM) return ERROR;
		G.vertices[i].data = V[i];   //关键字域赋值
		G.vertices[i].firstarc = NULL;   //弧指针域先置空
	}
	G.vexnum = i;
	for (i = 0; VR[i][0] + 1; i++)
	{
		int m = LocateVertex(V, VR[i][0]), n = LocateVertex(V, VR[i][1]);   //定位两个顶点
		if (!G.vertices[m].firstarc)   //首次赋值
		{
			G.vertices[m].firstarc = (ArcNode*)malloc(sizeof(ArcNode));
			G.vertices[m].firstarc->adjvex = n;
			G.vertices[m].firstarc->nextarc = NULL;
		}
		else   //首插法插入新的边关系
		{
			ArcNode* p = (ArcNode*)malloc(sizeof(ArcNode));
			p->adjvex = n;
			p->nextarc = G.vertices[m].firstarc;
			G.vertices[m].firstarc = p;
		}
		if (!G.vertices[n].firstarc)   //同理处理n到p的边
		{
			G.vertices[n].firstarc = (ArcNode*)malloc(sizeof(ArcNode));
			G.vertices[n].firstarc->adjvex = m;
			G.vertices[n].firstarc->nextarc = NULL;
		}
		else
		{
			ArcNode* p = (ArcNode*)malloc(sizeof(ArcNode));
			p->adjvex = m;
			p->nextarc = G.vertices[n].firstarc;
			G.vertices[n].firstarc = p;
		}
	}
	G.arcnum = i;
	//printf("arcnum:%d\n", i);
	return OK;
}

status DestroyGraph(ALGraph& G)
{
	for (int i = 0; i < G.vexnum; i++)
	{
		ArcNode* p = G.vertices[i].firstarc, * q;
		while (p)
		{
			q = p->nextarc;
			free(p);   //从前往后回收结点空间
			p = q;   //向后移动
		}
	}
	G.vexnum = 0;
	G.arcnum = 0;
	return OK;
}

int LocateVex(ALGraph G, KeyType u)
{
	for (int i = 0; i < G.vexnum; i++)
		if (G.vertices[i].data.key == u) return i;
	return -1;
}

status PutVex(ALGraph& G, KeyType u, VertexType value)
{
	int p;
	if ((p = LocateVex(G, u)) == -1) return ERROR;   //未找到u
	if ((u - value.key) && (LocateVex(G, value.key) + 1)) return ERROR;   //关键字重复
	G.vertices[p].data = value;
	return OK;
}

int FirstAdjVex(ALGraph G, KeyType u)
{
	int p;
	if ((p = LocateVex(G, u)) == -1) return INFEASIBLE;
	return G.vertices[p].firstarc->adjvex;
}

int NextAdjVex(ALGraph G, KeyType v, KeyType w)
{
	int p = LocateVex(G, v), q = LocateVex(G, w);
	if (!(p + 1) || !(q + 1)) return INFEASIBLE;
	ArcNode* r = G.vertices[p].firstarc;
	while (r->adjvex != q && r)
		r = r->nextarc;
	if (!r) return INFEASIBLE;
	if (!r->nextarc) return INFEASIBLE;
	return r->nextarc->adjvex;
}

status InsertVex(ALGraph& G, VertexType v)
{
	if (G.vexnum == MAX_VERTEX_NUM) return ERROR;   //图已满，不能插入
	if (LocateVex(G, v.key) + 1) return ERROR;    //关键字重复
	G.vertices[G.vexnum].data = v;
	G.vertices[G.vexnum].firstarc = NULL;
	G.vexnum++;
	return OK;
}

void ArcClear(ALGraph& G, int p)
{
	for (int i = 0; i < G.vexnum; i++)
	{
		if (!G.vertices[i].firstarc) continue;	  //没有邻接边，不处理
		if (G.vertices[i].firstarc->adjvex == p)   //首条邻接边需要删除，特判
		{
			ArcNode* del = G.vertices[i].firstarc;   //先保存要删除的结点地址
			G.vertices[i].firstarc = G.vertices[i].firstarc->nextarc;
			free(del);
			G.arcnum--;
			continue;   //由于不重复，开始下个顶点的处理
		}
		ArcNode* m, * n = NULL;
		m = G.vertices[i].firstarc;
		n = m->nextarc;
		while (n)
		{
			if (n->adjvex == p)
			{
				m->nextarc = n->nextarc;
				free(n);
				G.arcnum--;
			}
			m = m->nextarc;
			if (m)   //m为空时m->nextarc越界
				n = m->nextarc;
			else n = NULL;
		}
	}
}

void ArcRenew(ALGraph& G, int p)//因为邻接数组的移位操作，大于p的边需要更新指向
{
	for (int i = 0; i < G.vexnum; i++)
	{
		ArcNode* t = G.vertices[i].firstarc;
		while (t)
		{
			if (t->adjvex > p) t->adjvex--;
			t = t->nextarc;
		}
	}
}

status DeleteVex(ALGraph& G, KeyType v)
{
	int p = LocateVex(G, v);   //定位要删除的顶点
	if (p == -1) return ERROR;   //该顶点不存在
	if (G.vexnum == 1) return ERROR;    //不能为空图，故一个顶点时不能删
	ArcNode* t = G.vertices[p].firstarc, * q;
	while (t)   //清空要去除的边
	{
		q = t->nextarc;
		free(t);
		t = q;
	}
	for (int i = p; i < G.vexnum - 1; i++)
		G.vertices[i] = G.vertices[i + 1];   //后续顶点前移，相当于删除
	G.vexnum--;
	ArcClear(G, p);   //去除与v有关的弧
	ArcRenew(G, p);   //更新弧的指向
	return OK;
}

bool ArcCheck(ALGraph G, int p, int q)//检查G中是否已含有<p,q>这条边
{
	ArcNode* t = G.vertices[p].firstarc;
	while (t)
	{
		if (t->adjvex == q) return false;
		t = t->nextarc;
	}
	return true;
}

status InsertArc(ALGraph& G, KeyType v, KeyType w)
{
	int p = LocateVex(G, v), q = LocateVex(G, w);
	if (p == -1 || q == -1) return ERROR;
	if (!ArcCheck(G, p, q)) return ERROR;   //该边已存在，不需要插入
	ArcNode* PtoQ = (ArcNode*)malloc(sizeof(ArcNode));
	ArcNode* QtoP = (ArcNode*)malloc(sizeof(ArcNode));
	PtoQ->adjvex = q;
	PtoQ->nextarc = G.vertices[p].firstarc;
	G.vertices[p].firstarc = PtoQ;
	QtoP->adjvex = p;
	QtoP->nextarc = G.vertices[q].firstarc;
	G.vertices[q].firstarc = QtoP;
	G.arcnum++;
	return OK;
}

void DeleteArcHelper(ALGraph& G, int p, int q)//删除p到q的边
{
	if (G.vertices[p].firstarc->adjvex == q)   //首条邻接边需要删除，特判
	{
		ArcNode* del = G.vertices[p].firstarc;
		G.vertices[p].firstarc = G.vertices[p].firstarc->nextarc;
		free(del);
		G.arcnum--;
		return;
	}
	ArcNode* m, * n = NULL;
	m = G.vertices[p].firstarc;
	n = m->nextarc;
	while (n)
	{
		if (n->adjvex == q)
		{
			m->nextarc = n->nextarc;
			free(n);
			G.arcnum--;
		}
		m = m->nextarc;
		if (m)
			n = m->nextarc;
		else n = NULL;
	}
}

status DeleteArc(ALGraph& G, KeyType v, KeyType w)
{
	int p = LocateVex(G, v), q = LocateVex(G, w);
	if (p == -1 || q == -1) return ERROR;   //未找到顶点
	if (ArcCheck(G, p, q)) return ERROR;   //该边不存在，无法删除
	DeleteArcHelper(G, p, q);
	DeleteArcHelper(G, q, p);
	G.arcnum++;
	return OK;
}

void visit(VertexType v)
{
	printf("%d %s ", v.key, v.others);
}

void DFSTraverseHelper(ALGraph& G, int v, void (*visit)(VertexType))
{
	//v是起点编号，IsVisited[]是一个全局数组，作为顶点的访问标记，初始时所有的元素均为0，表示所有顶点都未被访问
	//因图中可能存在回路，当前经过的顶点在将来还可能再次经过，所以要对每个顶点进行标记，以免重复访问
	if (IsVisited[v] == 1) return;
	IsVisited[v] = 1;   //置已访问标记
	visit(G.vertices[v].data);
	ArcNode* p = G.vertices[v].firstarc;   //p指向顶点v的第一条边
	while (p)
	{
		if (IsVisited[p->adjvex] == 0)   //若顶点未访问，则递归访问它
			DFSTraverseHelper(G, p->adjvex, visit);
		p = p->nextarc;      //p指向顶点v的下一条边的终点
	}
}

status DFSTraverse(ALGraph& G, void (*visit)(VertexType))
{
	for (int i = 0; i < G.vexnum; i++)
	{
		DFSTraverseHelper(G, i, visit);
	}
	putchar('\n');
	return OK;
}

void BFSTraverseHelper(ALGraph& G, int v, void (*visit)(VertexType))
{
	if (IsVisited[v]) return;
	ArcNode* p;
	int que[maxSize], front = 0, rear = 0;   //队列的简单写法
	int j;
	visit(G.vertices[v].data);
	IsVisited[v] = 1;
	rear = (rear + 1) % maxSize;   //当前顶点入队
	que[rear] = v;
	while (front != rear)   //队空时说明遍历完成
	{
		front = (front + 1) % maxSize;   //顶点出队
		j = que[front];
		p = G.vertices[j].firstarc;   //p指向出队顶点j的第一条边
		while (p)   //将p的所有邻接点中未被访问的入队
		{
			if (IsVisited[p->adjvex] == 0)   //当前邻接结点未被访问，则入队
			{
				visit(G.vertices[p->adjvex].data);
				IsVisited[p->adjvex] = 1;
				rear = (rear + 1) % maxSize;   //该顶点入队
				que[rear] = p->adjvex;
			}
			p = p->nextarc;   //p指向j的下一条边
		}
	}
}

status BFSTraverse(ALGraph& G, void (*visit)(VertexType))
{
	for (int i = 0; i < G.vexnum; i++)
	{
		BFSTraverseHelper(G, i, visit);
	}
	putchar('\n');
	return OK;
}

status SaveGraph(ALGraph G, char FileName[])
{
	FILE* fp;
	if ((fp = fopen(FileName, "w")) == NULL) return ERROR;
	ArcNode* p;
	for (int i = 0; i < G.vexnum; i++)
	{
		fprintf(fp, "%d %s ", G.vertices[i].data.key, G.vertices[i].data.others);
		p = G.vertices[i].firstarc;
		while (p)
		{
			fprintf(fp, "%d ", p->adjvex);
			p = p->nextarc;
		}
		fprintf(fp, "-1\n");   //结尾标记
	}
	fclose(fp);
	return OK;
}

status LoadGraph(ALGraph& G, char FileName[])
{
	if (G.vexnum) return INFEASIBLE;
	G.vexnum = 0;
	G.arcnum = 0;
	FILE* fp;
	if ((fp = fopen(FileName, "r")) == NULL) return ERROR;
	int temp = 0;
	ArcNode* p, * tail;
	while (fscanf(fp, "%d", &G.vertices[G.vexnum].data.key) != EOF)
	{
		fscanf(fp, "%s", G.vertices[G.vexnum].data.others);
		fscanf(fp, "%d", &temp);
		if (temp + 1)
		{
			G.vertices[G.vexnum].firstarc = (ArcNode*)malloc(sizeof(ArcNode));
			G.vertices[G.vexnum].firstarc->adjvex = temp;
			G.vertices[G.vexnum].firstarc->nextarc = NULL;
			G.arcnum++;
		}
		else  G.vertices[G.vexnum].firstarc = NULL;
		fscanf(fp, "%d", &temp);
		tail = G.vertices[G.vexnum].firstarc;
		while ((temp + 1) && tail)   //判断tail，避免循环内越界
		{
			p = (ArcNode*)malloc(sizeof(ArcNode));
			p->adjvex = temp;
			p->nextarc = NULL;
			tail->nextarc = p;
			tail = p;
			G.arcnum++;
			fscanf(fp, "%d", &temp);
		}
		G.vexnum++;
	}
	G.arcnum /= 2;
	fclose(fp);
	return OK;
}

status MultipleGraphsInput(MultipleGraphs& Gs)
{
	int state;
	int j = 0, num = 0;
	printf("输入多个图及其内部数据：\n");
	printf("输入1开始！\n");
	while (scanf("%d", &state) == 1 && state == 1)
	{
		if (Gs.length >= 10)
		{
			printf("多图已满，不能再继续输入数据！");
			return OVERFLOW;
		}
		printf("请输入图的名称：\n");
		scanf("%s", Gs.elem[j].name);
		DestroyGraph(Gs.elem[j].G);
		int i = 0;
		printf("请输入顶点信息(先关键字再名称，空格隔开,eg.5 线性表)\n");
		do
		{
			scanf("%d%s", &V[i].key, V[i].others);
		} while (V[i++].key != -1);
		i = 0;
		printf("请输入关系对序列,空格隔开\n");
		do
		{
			scanf("%d%d", &VR[i][0], &VR[i][1]);
		} while (VR[i++][0] != -1);
		if (CreateGraph(Gs.elem[j].G, V, VR) == ERROR) printf("输入数据错误，无法创建");
		else if (Gs.elem[j].G.arcnum != i - 1)
		{
			printf("边的数目错误！\n");
			return ERROR;
		}
		j++;
		Gs.length++;
		printf("是否继续？1：继续输入；-1：结束输入！\n");
	}
	return OK;
}

int LocateG(MultipleGraphs Gs, char Gname[])  //寻找对应的图，返回对应线性表的位置下标
{
	int i, j;
	for (i = 0; i < Gs.length; i++)//遍历多个名称的线性表
	{
		for (j = 0; Gname[j] != 0 && Gs.elem[i].name[j] != 0; j++)//遍历各线性表的名称字符串
		{
			if (Gs.elem[i].name[j] != Gname[j])
				break;
		}
		if (Gname[j] == 0 && Gs.elem[i].name[j] == 0)
			return i + 1;
	}
	return 0;
}

status RemoveGraph(MultipleGraphs& Gs, char GraphName[])
{
	int i = LocateG(Gs, GraphName);
	ALGraph tem = Gs.elem[i].G;
	for (int j = i - 1; j <= Gs.length - 2; j++)
	{
		Gs.elem[j] = Gs.elem[j + 1];
	}
	DestroyGraph(tem);
	Gs.length--;
	return OK;
}

status InsertGraph(MultipleGraphs& Gs, int k, char GraphName[])
{
	if (Gs.length >= 10)
	{
		printf("多图已满，不能执行插入操作！");
		return OVERFLOW;
	}
	if (k<1 || k>Gs.length + 1) return ERROR;
	for (int j = Gs.length; j >= k; j--)
		Gs.elem[j] = Gs.elem[j - 1];
	int pos = k - 1;
	strcpy(Gs.elem[pos].name, GraphName);
	printf("请输入图的名称：\n");
	DestroyGraph(Gs.elem[pos].G);
	int i = 0;
	printf("请输入顶点信息(先关键字再名称，空格隔开,eg.5 线性表)\n");
	do
	{
		scanf("%d%s", &V[i].key, V[i].others);
	} while (V[i++].key != -1);
	i = 0;
	printf("请输入关系对序列,空格隔开\n");
	do
	{
		scanf("%d%d", &VR[i][0], &VR[i][1]);
	} while (VR[i++][0] != -1);
	if (CreateGraph(Gs.elem[pos].G, V, VR) == ERROR) printf("输入数据错误，无法创建");
	else if (G.arcnum != i - 1)
	{
		printf("边的数目错误！\n");
		return ERROR;
	}
	Gs.length++;
	return OK;
}

status ShowMultipleGraphs(MultipleGraphs& Gs)
{
	if (!Gs.length) return ERROR;
	int i;
	for (i = 0; i < Gs.length; i++)
	{
		printf("-------------------------------data---------------------------------\n");
		printf("名称：%s\n", Gs.elem[i].name);
		ShowGraph(Gs.elem[i].G);
		printf("-------------------------------end----------------------------------\n");
		putchar('\n');
	}
	return OK;
}

status InitMultipleGraphs(MultipleGraphs& Gs)
{
	for (int i = 0; i < Gs.length; i++)
		DestroyGraph(Gs.elem[i].G);
	Gs.length = 0;
	return OK;
}
*/
