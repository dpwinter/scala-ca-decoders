#ifndef CA_H
#define CA_H

#include <iostream>
#include <array>
#include <iostream>


template <size_t T>
class CA
{
	private:
		// int age;
		int ncells;
		std::array<int,T> cells;
	public:
		CA() : ncells{ T } { reset(); }
		void reset() 
		{ 
			cells.fill(0);
			// age = 0; 
		}
		void step(RepCode<T>& code, double q=0.0, double psig=0.0)
		{
			// age = (age + 1) % ncells; 
			// age = (age + 1) % int((ncells-1)/2 + (ncells-1)/4); // medium reset
			// age = (age + 1) % int((ncells-1)/2); // early reset
			// age = (age + 1) % int((ncells-1)/4); // very early reset
			// if (age == 0) 
			// 	reset();

			std::array<bool,T> syndrome = code.getSyndrome(q); // input
			std::array<int,T> next_cells {0}; // next gen (signals)

			for (int i=0; i<ncells; i++)
			{
				int i_prev = i==0 ? ncells-1 : i-1; 
				int i_next = i==ncells-1 ? 0 : i+1; 
				
				// syndromes
				bool sl = syndrome[i_prev];
				bool sc = syndrome[i];
				bool sr = syndrome[i_next];

				// states in neighborhood
				int stl = cells[i_prev];
				int stc = cells[i];
				int str = cells[i_next];

				// new cell state (l,r) -> r=1, l=2, rl=3
				bool li = (str&2) || (!(str&1) & !(str&2) & sr);
				bool ri = (stl&1) || (!(stl&1) & !(stl&2) & sl);


				// correction
				if ( sl & sc) {
					code.flipQubit(i_prev);
				}
				else if (!sl & sc & !sr & !li & ri) {
					code.flipQubit(i_prev);
				}
				else if (!sr & sc & !sl & li & !ri) {
					code.flipQubit(i);
				}
				// if ( (sl & sc) || ( !sl & sc & !li & ri )) // flip left
				// 	code.flipQubit(i_prev);
				// else if (!sr & sc & li & !ri)  // flip right
				// 	code.flipQubit(i);

				// noise on signals
				if (Random::getError(psig)) li = li ^ 1;
				if (Random::getError(psig)) ri = ri ^ 1;

				next_cells[i] = 2*li + ri; // construct new cell state

			}

			cells = next_cells;
		}

		friend std::ostream& operator<<(std::ostream &out, const CA<T>& ca)
		{
			for (int i=0; i<ca.ncells; i++)
			{
				// out << ca.cells[i] << ' '; // TODO bit-shifting for better display
				switch(ca.cells[i])
				{
					case 0:
						out << "  ";
						break;
					case 1:
						out << "> ";
						break;
					case 2:
						out << "< ";
						break;
					case 3:
						out << "\u2277 ";
						break;
				}
			}
			out << '\n';
			return out;
		}
};

#endif
